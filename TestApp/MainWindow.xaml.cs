using System;
using System.Threading;
using System.Windows;
using AudioPlugSharp;
using NAM.Plugin;

namespace NamApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NAMPlugin plugin;

        public MainWindow()
        {
            InitializeComponent();

            plugin = new NAMPlugin();

            plugin.Host = new DummyHost();
            plugin.Initialize();

            new Thread(new ThreadStart(RunAudio)).Start();
        }

        void RunAudio()
        {
            while (true)
            {
                plugin.Model.ProcessSample(0);
            }
        }
    }
}
