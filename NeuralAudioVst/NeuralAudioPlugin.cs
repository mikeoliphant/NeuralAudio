using System;
using System.Windows;
using System.Windows.Controls;
using AudioPlugSharp;
using AudioPlugSharpWPF;
using NeuralModel;

namespace NeuralAudioVst
{
    public class NeuralAudioPluginSaveState : AudioPluginWPFSaveState
    {
        public string ModelPath { get; set; }
    }

    public class NeuralAudioPlugin : AudioPluginWPF
    {
        AudioIOPort monoInput = null;
        AudioIOPort monoOutput = null;

        public NeuralModelConfig Model { get; private set; } = null;
        public string ModelPath { get; private set; } = null;

        public NeuralAudioPluginSaveState NeuralAudioPluginSaveState { get { return (SaveStateData as NeuralAudioPluginSaveState) ?? new NeuralAudioPluginSaveState(); } }

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

            SaveStateData = new NeuralAudioPluginSaveState();
        }

        public override void Initialize()
        {
            base.Initialize();

            InputPorts = new AudioIOPort[] { monoInput = new AudioIOPort("Mono Input", EAudioChannelConfiguration.Mono) };
            OutputPorts = new AudioIOPort[] { monoOutput = new AudioIOPort("Mono Output", EAudioChannelConfiguration.Mono) };

            AddParameter(new AudioPluginParameter
            {
                ID = "gain",
                Name = "Gain",
                Type = EAudioPluginParameterType.Float,
                MinValue = -20,
                MaxValue = 20,
                DefaultValue = 0,
                ValueFormat = "{0:0.0}dB"
            });

            AddParameter(new AudioPluginParameter
            {
                ID = "volume",
                Name = "Level",
                Type = EAudioPluginParameterType.Float,
                MinValue = -20,
                MaxValue = 20,
                DefaultValue = 0,
                ValueFormat = "{0:0.0}dB"
            });
        }

        public override UserControl GetEditorView()
        {
            return new EditorView();
        }

        public override byte[] SaveState()
        {
            NeuralAudioPluginSaveState.ModelPath = ModelPath;

            return base.SaveState();
        }

        public override void RestoreState(byte[] stateData)
        {
            base.RestoreState(stateData);

            if (!string.IsNullOrEmpty(NeuralAudioPluginSaveState.ModelPath))
            {
                try
                {
                    LoadModel(NeuralAudioPluginSaveState.ModelPath);
                }
                catch (Exception ex)
                {
                    Logger.Log("Failed to restore model: " + ex.ToString());
                }
            }
        }

        public void LoadModel(string path)
        {
            Logger.Log("Load Model: " + path);

            Model = NeuralModelConfig.FromFile(path);

            ModelPath = path;
        }

        public override void Process()
        {
            base.Process();

            Host.ProcessAllEvents();

            double gain = GetParameter("gain").ProcessValue;
            double linearGain = Math.Pow(10.0, 0.05 * gain);

            double vol = GetParameter("volume").ProcessValue;
            double linearVolume = Math.Pow(10.0, 0.05 * vol);

            ReadOnlySpan<double> inSamples = monoInput.GetAudioBuffer(0);
            Span<double> outSamples = monoOutput.GetAudioBuffer(0);

            if (Model == null)
            {
                monoInput.PassThroughTo(monoOutput);
            }
            else
            {
                for (int i = 0; i < inSamples.Length; i++)
                {
                    outSamples[i] = (double)Model.ProcessSample((float)(inSamples[i] * linearGain)) * linearVolume;
                }
            }
        }
    }

}
