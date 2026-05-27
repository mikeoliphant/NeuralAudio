namespace NeuralAudioTest
{
    using NeuralAudio;

    internal class Program
    {
        static void Main(string[] args)
        {
            NeuralModelLoader loader = new();

            loader.SetWaveNetModelLoadMode(EModelLoadMode.Internal);

            NeuralModel model = loader.CreateModelFromFile("BossWN-standard.nam");

            var input = new float[1024];
            var output = new float[1024];

            model.Process(input, output, 1024);
        }
    }
}
