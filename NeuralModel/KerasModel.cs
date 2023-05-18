using System.Text.Json;
using NeuralNet;

namespace NeuralModel
{
    public class KerasModelConfig : NeuralModelConfig
    {
        public static MatrixF MatrixFromJson(JsonElement element)
        {
            MatrixF matrix = new MatrixF(element.GetArrayLength(), element[0].GetArrayLength());

            for (int row = 0; row < matrix.NumRows; row++)
            {
                for (int col = 0; col < matrix.NumCols; col++)
                {
                    matrix[row, col] = element[row][col].GetSingle();
                }
            }

            return matrix;
        }

        public static KerasModelConfig FromJson(JsonDocument doc)
        {
            KerasModelConfig modelConfig = new KerasModelConfig();

            var layerConfigs = doc.RootElement.GetProperty("layers");

            int numLayers = layerConfigs.GetArrayLength();

            if (numLayers < 2)
            {
                throw new InvalidDataException("LSTM network must have at least one LSTM and one Dense layer");
            }

            List<LSTMLayer> layers = new List<LSTMLayer>();

            int lastLayerSize = 1;

            for (int i = 0; i < (numLayers - 1); i++)
            {
                var layerConfig = layerConfigs[i];

                if (!layerConfig.GetProperty("type").GetString().Equals("lstm", StringComparison.InvariantCultureIgnoreCase))
                {
                    throw new InvalidDataException("Layer " + i + " is not an LSTM layer");
                }

                int layerSize = layerConfig.GetProperty("shape")[2].GetInt32();

                var weights = layerConfig.GetProperty("weights");

                var inputWeights = MatrixFromJson(weights[0]);
                var hiddenWeights = MatrixFromJson(weights[1]);
                var bias = weights[2].Deserialize<float[]>();

                layers.Add(new LSTMLayer(lastLayerSize, layerSize, inputWeights, hiddenWeights, bias));

                lastLayerSize = layerSize;
            }

            var denseLayerConfig = layerConfigs[numLayers - 1];

            if (!denseLayerConfig.GetProperty("type").GetString().Equals("dense", StringComparison.InvariantCultureIgnoreCase))
            {
                throw new InvalidDataException("Last layer is not a Dense layer");
            }

            float[] headWeights = new float[lastLayerSize];

            var denseWeights = denseLayerConfig.GetProperty("weights");

            for (int i = 0; i < headWeights.Length; i++)
            {
                headWeights[i] = denseWeights[0][i][0].GetSingle();    // dense weights are stored as matrix of Nx1
            }

            bool doSkip = false;

            JsonElement skipElement;

            if (doc.RootElement.TryGetProperty("in_skip", out skipElement))
            {
                doSkip = (skipElement.GetInt32() == 1);
            }

            LSTMNetwork lstmNet = new LSTMNetwork(headWeights, denseWeights[1][0].GetSingle(), doSkip);

            lstmNet.Layers = layers;

            modelConfig.Network = lstmNet;


            return modelConfig;
        }
    }
}
