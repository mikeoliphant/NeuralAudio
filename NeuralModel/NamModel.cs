using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using NeuralNet;

namespace NeuralModel
{
    public class NamModelConfig : NeuralModelConfig
    {
        [JsonPropertyName("version")]
        public string Version { get; set; }
        [JsonPropertyName("architecture")]
        public string Architecture { get; set; }
        [JsonPropertyName("weights")]
        public float[] Weights { get; set; }

        public new static NamModelConfig FromFile(string filePath)
        {
            NamModelConfig model = null;

            try
            {
                using (FileStream stream = File.OpenRead(filePath))
                {
                    JsonDocument doc = JsonDocument.Parse(stream);
                    {
                        JsonElement elem;

                        if (doc.RootElement.TryGetProperty("architecture", out elem))
                        {
                            string architecture = elem.GetString();

                            switch (architecture)
                            {
                                case "LSTM":
                                    return NAMLSTMModelConfig.FromJson(doc);

                                case "WaveNet":
                                    return NAMWaveNetModelConfig.FromJson(doc);

                                default:
                                    throw new InvalidDataException("Unknown model architecture [" + architecture + "]");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("Error parsing model config: " + ex.Message);
            }

            throw new InvalidDataException("Error parsing model config");
        }
    }

    public class NAMLSTMModelConfig : NamModelConfig
    {
        [JsonPropertyName("config")]
        public NamLSTMConfig Config { get; set; }

        public static NAMLSTMModelConfig FromJson(JsonDocument doc)
        {
            NAMLSTMModelConfig model = doc.Deserialize<NAMLSTMModelConfig>();

            Span<float> weightSpan = new Span<float>(model.Weights);

            int offset = 0;
            int size;

            int gateSize = 4 * model.Config.HiddenSize;

            List<LSTMLayer> layers = new List<LSTMLayer>();

            for (int layer = 0; layer < model.Config.NumLayers; layer++)
            {
                int inputSize = (layer == 0) ? model.Config.InputSize : model.Config.HiddenSize;

                // NAM LSTM has input/hidden weights glommed together column-wise
                size = (4 * model.Config.HiddenSize) * (inputSize + model.Config.HiddenSize);
                var weights = weightSpan.Slice(offset, size).ToArray();
                offset += size;

                var inputWeights = new float[gateSize * inputSize];
                var hiddenWeights = new float[gateSize * model.Config.HiddenSize];

                // Separate the input/hidden weights
                for (int row = 0; row < gateSize; row++)
                {
                    int rowPos = row * (inputSize + model.Config.HiddenSize);

                    Array.Copy(weights, rowPos, inputWeights, row * inputSize, inputSize);
                    Array.Copy(weights, rowPos + inputSize, hiddenWeights, row * model.Config.HiddenSize, model.Config.HiddenSize);
                }

                size = gateSize;
                var bias = weightSpan.Slice(offset, size).ToArray();
                offset += size;

                // NAM provides initial hidden/cell state, but it doesn't really do anything so ignore it
                size = model.Config.HiddenSize;
                var hiddenState = weightSpan.Slice(offset, size);
                offset += size;

                size = model.Config.HiddenSize;
                var cellState = weightSpan.Slice(offset, size);
                offset += size;

                layers.Add(new LSTMLayer(inputSize, model.Config.HiddenSize, MatrixF.FromRowNormalData(inputWeights, 4 * model.Config.HiddenSize, inputSize),
                    MatrixF.FromRowNormalData(hiddenWeights, 4 * model.Config.HiddenSize, model.Config.HiddenSize), bias));
            }

            size = model.Config.HiddenSize;
            var headWeights = weightSpan.Slice(offset, size).ToArray();
            offset += size;

            LSTMNetwork lstmNet = new LSTMNetwork(headWeights, weightSpan[offset]);
            lstmNet.Layers = layers;

            model.Network = lstmNet;

            return model;
        }
    }

    public class NamLSTMConfig
    {
        [JsonPropertyName("input_size")]
        public int InputSize { get; set; }
        [JsonPropertyName("hidden_size")]
        public int HiddenSize { get; set; }
        [JsonPropertyName("num_layers")]
        public int NumLayers { get; set; }
    }

    public class NAMWaveNetModelConfig : NamModelConfig
    {
        [JsonPropertyName("config")]
        public NAMWaveNetConfig Config { get; set; }

        public static NAMWaveNetModelConfig FromJson(JsonDocument doc)
        {
            NAMWaveNetModelConfig model = doc.Deserialize<NAMWaveNetModelConfig>();

            ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>(model.Weights);

            int offset = 0;
            int size;

            List<WaveNetLayer> layers = new List<WaveNetLayer>();

            foreach (NAMWaveNetLayerConfig layerConfig in model.Config.Layers)
            {
                // Rechannel
                Conv1x1 rechannel = CreateConv1x1(layerConfig.InputSize, layerConfig.Channels, doBias: false, weightSpan, ref offset);

                List<WaveNetDilation> dilations = new List<WaveNetDilation>();

                // Dilations
                foreach (int dilation in layerConfig.Dilations)
                {
                    // out x in x kernel
                    int outChannels = layerConfig.Gated ? (layerConfig.Channels * 2) : layerConfig.Channels;

                    size = outChannels * layerConfig.Channels * layerConfig.KernelSize;

                    var convWeights = weightSpan.Slice(offset, size);
                    offset += size;

                    var convKernels = new MatrixF[layerConfig.KernelSize];

                    for (int k = 0; k < layerConfig.KernelSize; k++)
                    {
                        convKernels[k] = new MatrixF(outChannels, layerConfig.Channels);
                    }

                    int pos = 0;

                    for (int outChannel = 0; outChannel < outChannels; outChannel++)
                    {
                        for (int inChannel = 0; inChannel < layerConfig.Channels; inChannel++)
                        {
                            for (int k = 0; k < layerConfig.KernelSize; k++)
                            {
                                convKernels[k][outChannel, inChannel] = convWeights[pos++];
                            }
                        }
                    }

                    // Bias
                    size = outChannels;
                    var biasWeights = weightSpan.Slice(offset, size);
                    offset += size;

                    // MixIn
                    Conv1x1 mixIn = CreateConv1x1(layerConfig.ConditionSize, outChannels, doBias: false, weightSpan, ref offset);

                    // 1x1
                    Conv1x1 oneByOne = CreateConv1x1(layerConfig.Channels, layerConfig.Channels, doBias: true, weightSpan, ref offset);

                    WaveNetDilation dilationLayer = new WaveNetDilation(layerConfig.ConditionSize, outChannels, layerConfig.KernelSize, dilation, layerConfig.Activation, layerConfig.Gated, convKernels, mixIn, oneByOne);

                    dilations.Add(dilationLayer);
               }

                // Head Rechannel
                Conv1x1 headRechannel = CreateConv1x1(layerConfig.Channels, layerConfig.HeadSize, layerConfig.HeadBias, weightSpan, ref offset);

                WaveNetLayer layer = new WaveNetLayer(rechannel, headRechannel);
                layer.Dilations = dilations;

                layers.Add(layer);
            }

            // Last weight is just head scale, which is redundant since it is specified in the config
            float headScale = weightSpan[offset];

            WaveNetNetwork network = new WaveNetNetwork(model.Config.HeadScale);
            network.Layers = layers;

            model.Network = network;

            return model;
        }

        public static Conv1x1 CreateConv1x1(int inChannels, int outChannels, bool doBias, ReadOnlySpan<float> weights, ref int offset)
        {
            int size = outChannels * inChannels;

            MatrixF weightMatrix = MatrixF.FromRowNormalData(weights.Slice(offset, size), outChannels, inChannels);
            offset += size;

            if (doBias)
            {
                size = outChannels;

                var biasWeights = weights.Slice(offset, size);
                offset += size;

                return new Conv1x1(inChannels, outChannels, weightMatrix, biasWeights.ToArray());
            }

            return new Conv1x1(inChannels, outChannels, weightMatrix);
        }
    }

    public class NAMWaveNetConfig
    {
        [JsonPropertyName("layers")]
        public List<NAMWaveNetLayerConfig> Layers { get; set; }
        [JsonPropertyName("head_scale")]
        public float HeadScale { get; set; }       
    }

    public class NAMWaveNetLayerConfig
    {
        [JsonPropertyName("input_size")]
        public int InputSize { get; set; }
        [JsonPropertyName("condition_size")]
        public int ConditionSize { get; set; }
        [JsonPropertyName("channels")]
        public int Channels { get; set; }
        [JsonPropertyName("head_size")]
        public int HeadSize { get; set; }
        [JsonPropertyName("kernel_size")]
        public int KernelSize { get; set; }
        [JsonPropertyName("dilations")]
        public List<int> Dilations { get; set; }
        [JsonPropertyName("activation")]
        public string Activation { get; set; }
        [JsonPropertyName("gated")]
        public bool Gated { get; set; }
        [JsonPropertyName("head_bias")]
        public bool HeadBias { get; set; }
    }
}