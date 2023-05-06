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

        public List<LSTMLayer> Layers { get; set; } = new List<LSTMLayer>();

        public LSTMModel(int hiddenSize, float[] headWeights, float headBias)
        {
            this.HiddenSize = hiddenSize;

            this.headWeights = headWeights;
        }

        public void Process(float[] input, float[] output)
        {
            Layers[0].Process(input);

            for (int layer = 1; layer < Layers.Count; layer++)
            {
                Layers[layer].Process(Layers[layer - 1].HiddenState);
            }
            
            output[0] = VecOp.Dot(headWeights, Layers[Layers.Count - 1].HiddenState) + headBias;
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

        public LSTMLayer(int inputSize, int hiddenSize, float[] inputWeights, float[] hiddenWeights, float[] bias, float[] hiddenState, float[] cellState)
        {
            this.InputSize = inputSize;
            this.HiddenSize = hiddenSize;

            this.inputWeights = MatrixF.FromRowNormalData(inputWeights, 4 * HiddenSize, inputSize);
            this.hiddenWeights = MatrixF.FromRowNormalData(hiddenWeights, 4 * HiddenSize, HiddenSize);
            this.bias = bias;
            this.hiddenState = hiddenState;
            this.cellState = cellState;

            this.ifgo = new float[4 * HiddenSize];
        }

        public void Process(float[] input)
        {
            inputWeights.Mult(input, ifgo);
            hiddenWeights.MultAcc(hiddenState, ifgo);
            VecOp.Add(ifgo, bias);

            int i_offset = 0;
            int f_offset = HiddenSize;
            int g_offset = 2 * HiddenSize;
            int o_offset = 3 * HiddenSize;

            for (int i = 0; i < HiddenSize; i++)
            {
                cellState[i] = Activation.FastSigmoid(ifgo[i + f_offset]) * cellState[i] + Activation.FastSigmoid(ifgo[i + i_offset]) * Activation.FastTanh(ifgo[i + g_offset]);
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                hiddenState[i] = Activation.FastSigmoid(ifgo[i + o_offset]) * Activation.FastTanh(cellState[i]);
            }
        }
    }
}
