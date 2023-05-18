using System.Text.Json;
using NeuralNet;

namespace NeuralModel
{
    public class CoreAudioMLConfig : NeuralModelConfig
    {
        public new static CoreAudioMLConfig FromJson(JsonDocument doc)
        {
            CoreAudioMLConfig model = new CoreAudioMLConfig();

            var modelData = doc.RootElement.GetProperty("model_data");

            if (!modelData.GetProperty("model").GetString().Equals("SimpleRNN", StringComparison.InvariantCultureIgnoreCase))
            {
                throw new InvalidDataException("Only SimpleRNN models are supported");
            }

            if (!modelData.GetProperty("unit_type").GetString().Equals("LSTM", StringComparison.InvariantCultureIgnoreCase))
            {
                throw new InvalidDataException("Only SimpleRNN models are supported");
            }

            int numLayers = modelData.GetProperty("num_layers").GetInt32();
            int hiddenSize = modelData.GetProperty("hidden_size").GetInt32();

            var stateDict = doc.RootElement.GetProperty("state_dict");

            List<LSTMLayer> layers = new List<LSTMLayer>();

            int lastLayerSize = 1;

            for (int i = 0; i < numLayers; i++)
            {
                var inputWeights = stateDict.GetProperty("rec.weight_ih_l" + i);

                MatrixF inputMatrix = new MatrixF(4 * hiddenSize, lastLayerSize);

                for (int row = 0; row < (4 * hiddenSize); row++)
                {
                    for (int col = 0; col < lastLayerSize; col++)
                    {
                        inputMatrix[row, col] = inputWeights[row][col].GetSingle();
                    }
                }

                var hiddenWeights = stateDict.GetProperty("rec.weight_hh_l" + i);

                MatrixF hiddenMatrix = new MatrixF(4 * hiddenSize, hiddenSize);

                for (int row = 0; row < (4 * hiddenSize); row++)
                {
                    for (int col = 0; col < hiddenSize; col++)
                    {
                        hiddenMatrix[row, col] = hiddenWeights[row][col].GetSingle();
                    }
                }

                var inputBias = stateDict.GetProperty("rec.bias_ih_l" + i);
                var hiddenBias = stateDict.GetProperty("rec.bias_hh_l" + i);

                float[] bias = new float[4 * hiddenSize];

                for (int pos = 0; pos < bias.Length; pos++)
                {
                    bias[pos] = inputBias[pos].GetSingle() + hiddenBias[pos].GetSingle();
                }

                layers.Add(new LSTMLayer(lastLayerSize, hiddenSize, inputMatrix, hiddenMatrix, bias));

                lastLayerSize = hiddenSize;
            }

            var denseWeights = stateDict.GetProperty("lin.weight")[0];

            bool doSkip = false;

            JsonElement skipElement;

            if (modelData.TryGetProperty("skip", out skipElement))
            {
                doSkip = (skipElement.GetInt32() == 1);
            }

            LSTMNetwork lstmNetwork = new LSTMNetwork(denseWeights.Deserialize<float[]>(), stateDict.GetProperty("lin.bias")[0].GetSingle(), doSkip);

            lstmNetwork.Layers = layers;

            model.Network = lstmNetwork;

            return model;
        }
    }
}
