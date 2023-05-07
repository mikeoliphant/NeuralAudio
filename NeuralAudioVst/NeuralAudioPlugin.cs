using System;
using System.Windows.Controls;
using AudioPlugSharp;
using AudioPlugSharpWPF;
using NAM;

namespace NeuralAudioVst
{
    public class NeuralAudioPlugin : AudioPluginWPF
    {
        AudioIOPort monoInput = null;
        AudioIOPort monoOutput = null;

        public Model Model { get; private set; } = null;

        public NeuralAudioPlugin()
        {
            Company = "Mike Oliphant";
            Website = "nostatic.org";
            Contact = "contact@my.email";
            PluginName = "Neural Audio";
            PluginCategory = "Fx";
            PluginVersion = "1.0.0";

            // Unique 64bit ID for the plugin
            PluginID = 0x9248DF0699594851;

            HasUserInterface = true;
            EditorWidth = 200;
            EditorHeight = 100;
        }

        public override void Initialize()
        {
            base.Initialize();

            InputPorts = new AudioIOPort[] { monoInput = new AudioIOPort("Mono Input", EAudioChannelConfiguration.Mono) };
            OutputPorts = new AudioIOPort[] { monoOutput = new AudioIOPort("Mono Output", EAudioChannelConfiguration.Mono) };

            AddParameter(new AudioPluginParameter
            {
                ID = "gain",
                Name = "Input Gain",
                Type = EAudioPluginParameterType.Float,
                MinValue = -20,
                MaxValue = 20,
                DefaultValue = 0,
                ValueFormat = "{0:0.0}dB"
            });

            AddParameter(new AudioPluginParameter
            {
                ID = "volume",
                Name = "Output Volume",
                Type = EAudioPluginParameterType.Float,
                MinValue = -20,
                MaxValue = 20,
                DefaultValue = 0,
                ValueFormat = "{0:0.0}dB"
            });

            LoadModel(@"C:\Users\oliph\Downloads\model-1-12.nam");
        }

        public override UserControl GetEditorView()
        {
            return new EditorView();
        }

        void LoadModel(string path)
        {
            Model = Model.FromFile(path);
        }

        public override void Process()
        {
            base.Process();

            Host.ProcessAllEvents();

            double gain = GetParameter("gain").ProcessValue;
            double linearGain = Math.Pow(10.0, 0.05 * gain);

            double vol = GetParameter("volume").ProcessValue;
            double linearVolume = Math.Pow(10.0, 0.05 * vol);

            monoInput.ReadData();

            double[] inSamples = monoInput.GetAudioBuffers()[0];
            double[] outSamples = monoOutput.GetAudioBuffers()[0];

            for (int i = 0; i < inSamples.Length; i++)
            {
                outSamples[i] = (double)Model.ProcessSample((float)(inSamples[i] * linearGain)) * linearVolume;
            }

            monoOutput.WriteData();
        }
    }
}
