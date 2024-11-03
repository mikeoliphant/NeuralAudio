#include <RTNeural/RTNeural.h>
#include <NeuralAudio/NeuralModel.h>

int main()
{
	 auto model = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\MODOrange\\AMP Orange Nasty.json");

	NeuralAudio::NeuralModel::SetPreferNAM(true);

	auto namModel = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\BossLSTM-1x16.nam");

	NeuralAudio::NeuralModel::SetPreferNAM(false);

	auto rtNeuralModel = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\BossLSTM-1x16.nam");

	const int dataSize = 2048;

	std::vector<float> audioInput;
	audioInput.resize(dataSize);

	for (size_t n = 0; n < audioInput.size(); ++n)
		audioInput[n] = std::sin(3.14 * n * 0.01);

	std::vector<float> namOutput;
	std::vector<float> rtNeuralOutput;

	namOutput.resize(dataSize);
	rtNeuralOutput.resize(dataSize);

	float err = 0;

	for (int run = 0; run < 10; run++)
	{
		namModel->Process(audioInput.data(), namOutput.data(), dataSize);
		rtNeuralModel->Process(audioInput.data(), rtNeuralOutput.data(), dataSize);

		err = 0;

		for (size_t n = 0; n < dataSize; ++n)
		{
			float diff = namOutput[n] - rtNeuralOutput[n];

			err += (diff * diff);
		}
	}

	std::cout << "Tot err is: " << sqrt(err);

	return 0;
}