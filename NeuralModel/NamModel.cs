using System.Text.Json;
using System.Text.Json.Serialization;
using NeuralModel;
using NeuralNet;

namespace NeuralModel
{
    public class NamModelConfig : NeuralModelConfig
    {
        [JsonPropertyName("version")]
        public string Version { get; set; }
        [JsonPropertyName("architecture")]
        public string Architecture { get; set; }
        [JsonPropertyName("config")]
        public NamLSTMConfig Config { get; set; }
        [JsonPropertyName("weights")]
        public float[] Weights { get; set; }

        public LSTMModel NNModel { get; private set; }

        public static NamModelConfig FromFile(string filePath)
        {
            NamModelConfig model = null;

            using (FileStream stream = File.OpenRead(filePath))
            {
                model = JsonSerializer.Deserialize<NamModelConfig>(stream);
            }

            switch (model.Architecture)
            {
                case "LSTM":
                    try
                    {
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
                            var hiddenState = weightSpan.Slice(offset, size).ToArray();
                            offset += size;

                            size = model.Config.HiddenSize;
                            var cellState = weightSpan.Slice(offset, size).ToArray();
                            offset += size;

                            layers.Add(new LSTMLayer(inputSize, model.Config.HiddenSize, inputWeights, hiddenWeights, bias));
                        }

                        size = model.Config.HiddenSize;
                        var headWeights = weightSpan.Slice(offset, size).ToArray();
                        offset += size;

                        model.NNModel = new LSTMModel(model.Config.HiddenSize, headWeights, weightSpan[offset]);
                        model.NNModel.Layers = layers;
                    }
                    catch (Exception ex)
                    {
                        throw new InvalidDataException("Error parsing model config");
                    }

                    break;

                default:
                    throw new InvalidDataException("Unknown model architecture [" + model.Architecture + "]");
            }

            return model;
        }

        public override float ProcessSample(float sample)
        {
            return NNModel.Process(sample);
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
        public int Channels { get; set; }
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