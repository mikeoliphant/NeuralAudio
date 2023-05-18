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
        public Network Network { get; set; }

        public static NeuralModelConfig FromFile(string filePath)
        {
            string extention = Path.GetExtension(filePath);

            if (extention.Equals(".nam", StringComparison.InvariantCultureIgnoreCase))
            {
                return NamModelConfig.FromFile(filePath);
            }
            else if (extention.Equals(".json", StringComparison.InvariantCultureIgnoreCase))
            {
                try
                {
                    using (FileStream stream = File.OpenRead(filePath))
                    {
                        JsonDocument doc = JsonDocument.Parse(stream);

                        JsonElement elem;

                        if (doc.RootElement.TryGetProperty("layers", out elem))
                        {
                            return KerasModelConfig.FromJson(doc);
                        }

                        if (doc.RootElement.TryGetProperty("model_data", out elem))
                        {
                            return CoreAudioMLConfig.FromJson(doc);
                        }

                        throw new InvalidDataException("Model json data not in a recognized format");
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException("Unable to parse model json data: " + ex.Message);
                }
            }

            throw new InvalidOperationException("Unknown model file extenstion: " + extention);
        }

        public virtual float ProcessSample(float sample)
        {
            return Network.Process(sample);
        }
    }
}