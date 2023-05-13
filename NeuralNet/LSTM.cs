using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNet
{
    public class LSTMModel
    {
        public int HiddenSize { get; private set; }

        float[] headWeights;
        float headBias;
        float[] layerInput = new float[1];

        public List<LSTMLayer> Layers { get; set; } = new List<LSTMLayer>();

        public LSTMModel(int hiddenSize, float[] headWeights, float headBias)
        {
            this.HiddenSize = hiddenSize;

            this.headWeights = headWeights;
        }

        public float Process(float input)
        {
            layerInput[0] = input;

            Layers[0].Process(layerInput);

            for (int layer = 1; layer < Layers.Count; layer++)
            {
                Layers[layer].Process(Layers[layer - 1].HiddenState);
            }
            
            return VecOp.Dot(headWeights, Layers[Layers.Count - 1].HiddenState) + headBias;
        }
    }

    public class LSTMLayer
    {
        public int InputSize { get; private set; }
        public int HiddenSize { get; private set; }
        public float[] HiddenState
        {
            get { return hiddenState; }
        }

        MatrixF inputWeights;
        MatrixF hiddenWeights;
        float[] bias;
        float[] hiddenState;
        float[] ifgo;
        float[] cellState;

        public LSTMLayer(int inputSize, int hiddenSize, float[] inputWeights, float[] hiddenWeights, float[] bias)
        {
            this.InputSize = inputSize;
            this.HiddenSize = hiddenSize;

            this.inputWeights = MatrixF.FromRowNormalData(inputWeights, 4 * HiddenSize, inputSize);
            this.hiddenWeights = MatrixF.FromRowNormalData(hiddenWeights, 4 * HiddenSize, HiddenSize);
            this.bias = bias;
            this.hiddenState = new float[HiddenSize];
            this.cellState = new float[HiddenSize];

            this.ifgo = new float[4 * HiddenSize];
        }

        public void Process(ReadOnlySpan<float> input)
        {
            // Input and Hidden weights could be multiplied together in one pass, which is faster - but this is clearer
            inputWeights.Mult(input, ifgo);
            hiddenWeights.MultAcc(hiddenState, ifgo);
            VecOp.Add(ifgo, bias);

            int iOffset = 0;
            int fOffset = HiddenSize;
            int gOffset = 2 * HiddenSize;
            int oOffset = 3 * HiddenSize;

            for (int i = 0; i < HiddenSize; i++)
            {
                cellState[i] = Activation.FastSigmoid(ifgo[i + fOffset]) * cellState[i] + Activation.FastSigmoid(ifgo[i + iOffset]) * Activation.FastTanh(ifgo[i + gOffset]);
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                hiddenState[i] = Activation.FastSigmoid(ifgo[i + oOffset]) * Activation.FastTanh(cellState[i]);
            }
        }
    }
}
