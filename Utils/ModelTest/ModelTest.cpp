#include <RTNeural/RTNeural.h>
#include <NeuralAudio/NeuralModel.h>

static double BenchModel(NeuralAudio::NeuralModel* model, int blockSize, int numBlocks)
{
	std::vector<float> inData;
	inData.resize(blockSize);

	std::vector<float> outData;
	outData.resize(blockSize);

	auto start = std::chrono::high_resolution_clock::now();

	for (int block = 0; block < numBlocks; block++)
	{
		model->Process(inData.data(), outData.data(), blockSize);
	}

	auto end = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();
}

int main()
{
	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto wnStandardModelRTNeural = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\AppData\\Roaming\\stompbox\\NAM\\JCM2000Crunch.nam");

	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);

	auto wnStandardModelNAM = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\AppData\\Roaming\\stompbox\\NAM\\JCM2000Crunch.nam");

	double rt = BenchModel(wnStandardModelRTNeural, 64, 1024);
	double nam = BenchModel(wnStandardModelNAM, 64, 1024);

	std::cout << "RTNeural: " << rt << std::endl;
	std::cout << "Nam: " << nam << std::endl;
	std::cout << "RTNeural is: " << (nam / rt) << "x" << std::endl;

	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto wnFeatherModel = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\BossWN-feather.nam");

	auto model = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\MODOrange\\AMP Orange Nasty.json");

	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);

	auto namModel = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\BossLSTM-1x16.nam");

	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto rtNeuralModel = NeuralAudio::NeuralModel::CreateFromFile("C:\\Users\\oliph\\Downloads\\BossLSTM-1x16.nam");

	float db = rtNeuralModel->GetRecommendedOutputDBAdjustment();

	const int dataSize = 2048;

	std::vector<float> audioInput;
	audioInput.resize(dataSize);

	for (size_t n = 0; n < audioInput.size(); ++n)
		audioInput[n] = (float)std::sin(3.14 * n * 0.01);

	std::vector<float> namOutput;
	std::vector<float> rtNeuralOutput;

	namOutput.resize(dataSize);
	rtNeuralOutput.resize(dataSize);

	//float err = 0;

	//for (int run = 0; run < 10; run++)
	//{
	//	namModel->Process(audioInput.data(), namOutput.data(), dataSize);
	//	rtNeuralModel->Process(audioInput.data(), rtNeuralOutput.data(), dataSize);

	//	err = 0;

	//	for (size_t n = 0; n < dataSize; ++n)
	//	{
	//		float diff = namOutput[n] - rtNeuralOutput[n];

	//		err += (diff * diff);
	//	}
	//}

	//std::cout << "Tot err is: " << sqrt(err);



	return 0;
}
