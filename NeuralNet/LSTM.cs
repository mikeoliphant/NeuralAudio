using System;
using System.Collections.Generic;


namespace NeuralNet
{
    public class LSTMNetwork : Network
    {
        float[] headWeights;
        float headBias;
        float[] layerInput = new float[1];
        bool doSkip = false;

        public List<LSTMLayer> Layers { get; set; } = new List<LSTMLayer>();

        public LSTMNetwork(float[] headWeights, float headBias, bool doSkip = false)
        {
            this.headWeights = headWeights;
            this.headBias = headBias;
            this.doSkip = doSkip;
        }

        public override float Process(float input)
        {
            layerInput[0] = input;

            Layers[0].Process(layerInput);

            for (int layer = 1; layer < Layers.Count; layer++)
            {
                Layers[layer].Process(Layers[layer - 1].HiddenState);
            }

            float output = VecOp.Dot(headWeights, Layers[Layers.Count - 1].HiddenState) + headBias;

            return doSkip ? (input + output) : output;
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

        public LSTMLayer(int inputSize, int hiddenSize, MatrixF inputWeights, MatrixF hiddenWeights, float[] bias)
        {
            this.InputSize = inputSize;
            this.HiddenSize = hiddenSize;

            this.inputWeights = inputWeights;
            this.hiddenWeights = hiddenWeights;
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
