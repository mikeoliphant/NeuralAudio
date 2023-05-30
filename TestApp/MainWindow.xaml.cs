using System;
using System.Threading;
using System.Windows;
using AudioPlugSharpWPF;
using NeuralAudioVst;

namespace NamApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NeuralAudioPlugin plugin;

        public MainWindow()
        {
            InitializeComponent();

            plugin = new NeuralAudioPlugin();

            plugin.Host = new DummyHost();
            plugin.Initialize();

            plugin.LoadModel(@"C:\Users\oliph\Downloads\JCM2000Crunch.nam");

            EditorView.DataContext = plugin;

            new Thread(new ThreadStart(RunAudio)).Start();
        }

        void RunAudio()
        {
            while (true)
            {
                if (plugin.Model != null)
                    plugin.Model.ProcessSample(0);
            }
        }
    }
}
