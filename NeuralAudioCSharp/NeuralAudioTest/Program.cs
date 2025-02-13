namespace NeuralAudioTest
{
    using NeuralAudio;

    internal class Program
    {
        static void Main(string[] args)
        {
            NeuralModel model = NeuralModel.FromFile("BossWN-standard.nam");

            var input = new float[1024];
            var output = new float[1024];

            model.Process(input, output, 1024);
        }
    }
}
