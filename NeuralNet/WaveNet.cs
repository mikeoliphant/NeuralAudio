using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class Conv1x1
    {
        public int InputChannels { get; private set; }
        public int OutputChannels { get; private set; }
        public bool DoBias { get; private set; }

        MatrixF weights;
        float[]? bias = null;

        public Conv1x1(int inputChannels, int outputChannels, MatrixF weights)
        {
            this.InputChannels = inputChannels;
            this.OutputChannels = outputChannels;

            this.weights = weights;

            DoBias = false;
        }

        public Conv1x1(int inputChannels, int outputChannels, MatrixF weights, float[] bias)
            : this(inputChannels, outputChannels, weights)
        {
            this.bias = bias;

            DoBias = true;
        }

        public void Process(ref MatrixF input, ref MatrixF output)
        {
            weights.Mult(ref input, ref output);

            //if (DoBias)
            //{
            //}
            //else
            //{

            //}
        }

        public override string ToString()
        {
            return "Conv1x1 - " + InputChannels + "->" + OutputChannels + "]" + (DoBias ? "(biased)" : "");
        }
    }

    public class WaveNetDilation
    {
        public int ConditionSize { get; private set; }
        public int Channels { get; private set; }
        public int KernelSize { get; private set; }
        public int Dilation { get; private set; }
        public bool Gated { get; private set; }

        MatrixF[] convKernels;
        Conv1x1 mixIn;
        Conv1x1 oneByOne;

        public WaveNetDilation(int conditionSize, int channels, int kernelSize, int dilation, string activation, bool gated, MatrixF[] convKernels, Conv1x1 mixIn, Conv1x1 oneByOne)
        {
            this.ConditionSize = conditionSize;
            this.Channels = channels;
            this.KernelSize = kernelSize;
            this.Dilation = dilation;
            this.Gated = gated;

            this.convKernels = convKernels;
            this.mixIn = mixIn;
            this.oneByOne = oneByOne;
        }

        public override string ToString()
        {
            return "Dilation[" + Dilation + "]";
        }
    }

    public class WaveNetLayer
    {
        public List<WaveNetDilation> Dilations { get; set; } = new List<WaveNetDilation>();

        Conv1x1 rechannel;
        Conv1x1 headRechannel;

        public WaveNetLayer(Conv1x1 rechannel, Conv1x1 headRechannel)
        {
            this.rechannel = rechannel;
            this.headRechannel = headRechannel;
        }
    }

    public class WaveNetNetwork : Network
    {
        public List<WaveNetLayer> Layers { get; set; } = new List<WaveNetLayer>();

        float headScale;

        public WaveNetNetwork(float headScale)
        {
            this.headScale = headScale;
        }
    }
}
