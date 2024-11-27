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
	NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto wnStandardModelRTNeural = NeuralAudio::NeuralModel::CreateFromFile(std::string{ MODEL_DIR } + "BossWN-standard.nam");

	NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);

	auto wnStandardModelNAM = NeuralAudio::NeuralModel::CreateFromFile(std::string{ MODEL_DIR } + "BossWN-standard.nam");

	double rt = BenchModel(wnStandardModelRTNeural, 64, 1024);
	double nam = BenchModel(wnStandardModelNAM, 64, 1024);

	std::cout << "NAM Test" << std::endl;

	std::cout << "RTNeural: " << rt << std::endl;
	std::cout << "Nam: " << nam << std::endl;
	std::cout << "RTNeural is: " << (nam / rt) << "x" << std::endl;


	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto lstmModelRTNeural = NeuralAudio::NeuralModel::CreateFromFile(std::string{ MODEL_DIR } + "BossLSTM-1x16.nam");

	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);

	auto lstmModelNAM = NeuralAudio::NeuralModel::CreateFromFile(std::string{ MODEL_DIR } + "BossLSTM-1x16.nam");

	rt = BenchModel(lstmModelRTNeural, 64, 1024);
	nam = BenchModel(lstmModelNAM, 64, 1024);

	std::cout << "LSTM Test" << std::endl;

	std::cout << "RTNeural: " << rt << std::endl;
	std::cout << "Nam: " << nam << std::endl;
	std::cout << "RTNeural is: " << (nam / rt) << "x" << std::endl;

	return 0;
}
