using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using NeuralNet;

namespace NeuralModel
{
    public class NeuralModelConfig
    {
        public static NeuralModelConfig FromFile(string filePath)
        {
            string extention = Path.GetExtension(filePath);

            if (extention.Equals("nam", StringComparison.InvariantCultureIgnoreCase))
            {
                return NamModelConfig.FromFile(filePath);
            }
            else if (extention.Equals("json", StringComparison.InvariantCultureIgnoreCase))
            {
                return null;
            }

            throw new InvalidOperationException("Unknown model file extenstion: " + extention);
        }

        public virtual float ProcessSample(float sample)
        {
            return sample;
        }
    }
}