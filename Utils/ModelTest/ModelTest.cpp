#include <filesystem>
#include <iostream>
#include <NeuralAudio/NeuralModel.h>

static std::tuple<double, double> BenchModel(NeuralAudio::NeuralModel* model, int blockSize, int numBlocks)
{
	std::vector<float> inData;
	inData.resize(blockSize);

	std::vector<float> outData;
	outData.resize(blockSize);

	auto start = std::chrono::high_resolution_clock::now();

	double maxBlock = 0;

	for (int block = 0; block < numBlocks; block++)
	{
		auto blockStart = std::chrono::high_resolution_clock::now();

		model->Process(inData.data(), outData.data(), blockSize);

		auto blockEnd = std::chrono::high_resolution_clock::now();

		maxBlock = std::max(maxBlock, std::chrono::duration_cast<std::chrono::duration<double>> (blockEnd - blockStart).count());
	}

	auto end = std::chrono::high_resolution_clock::now();

	double tot = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();

	return std::tie(tot, maxBlock);
}

static double ComputeError(NeuralAudio::NeuralModel* model1, NeuralAudio::NeuralModel* model2, int blockSize, int numBlocks)
{
	std::vector<float> inData;
	inData.resize(blockSize);

	std::vector<float> outData;
	outData.resize(blockSize);

	std::vector<float> outData2;
	outData2.resize(blockSize);

	// Run zeros through for a bit to make sure both models are reset

	std::fill(inData.begin(), inData.end(), 0);

	int blocks = std::max(4096 / blockSize, 1);

	for (int block = 0; block < blocks; block++)
	{
		model1->Process(inData.data(), outData.data(), blockSize);
		model2->Process(inData.data(), outData2.data(), blockSize);
	}

	double totErr = 0;

	long pos = 0;

	for (int block = 0; block < numBlocks; block++)
	{
		for (int i = 0; i < blockSize; i++)
		{
			inData[i] = sin(pos++ * 0.01);
		}

		model1->Process(inData.data(), outData.data(), blockSize);
		model2->Process(inData.data(), outData2.data(), blockSize);

		for (int i = 0; i < blockSize; i++)
		{
			double diff = outData[i] - outData2[i];

			totErr += (diff * diff);
		}
	}

	return sqrt(totErr / (double)(blockSize * numBlocks));
}

void RunTests(std::filesystem::path modelPath)
{
	int dataSize = 4096 * 64;

	int blockSize = 64;
	int numBlocks = dataSize / blockSize;

	NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(blockSize);

	NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);
	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferRTNeural);

	auto rtNeuralModel = NeuralAudio::NeuralModel::CreateFromFile(modelPath);

	NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);
	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferNAMCore);

	auto namCoreModel = NeuralAudio::NeuralModel::CreateFromFile(modelPath);

	NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::ModelLoadMode::PreferInternal);
	NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::ModelLoadMode::PreferInternal);

	auto internalModel = NeuralAudio::NeuralModel::CreateFromFile(modelPath);

	double mse = ComputeError(namCoreModel, internalModel, blockSize, numBlocks);
	std::cout << "NAM vs Internal MSE: " << mse << std::endl;

	mse = ComputeError(namCoreModel, rtNeuralModel, blockSize, numBlocks);
	std::cout << "NAM vs RTNeural MSE: " << mse << std::endl;
	std::cout << std::endl;

	auto internal = BenchModel(internalModel, blockSize, numBlocks);
	auto rt = BenchModel(rtNeuralModel, blockSize, numBlocks);
	auto nam = BenchModel(namCoreModel, blockSize, numBlocks);

	std::cout << "NAM Core: " << std::get<0>(nam) << " (" << std::get<1>(nam) << ")" << std::endl;
	std::cout << "RTNeural: " << std::get<0>(rt) << " (" << std::get<1>(rt) << ")" << std::endl;
	std::cout << "Internal: " << std::get<0>(internal) << " (" << std::get<1>(internal) << ")" << std::endl;
	std::cout << "RTNeural is: " << (std::get<0>(nam) / std::get<0>(rt)) << "x NAM" << std::endl;
	std::cout << "Internal is: " << (std::get<0>(nam) / std::get<0>(internal)) << "x NAM" << std::endl;

	std::cout << std::endl;
}

int RunDefaultTests()
{
	std::filesystem::path modelPath = std::filesystem::current_path();

	while (modelPath.filename() != "Utils")
	{
		modelPath = modelPath.parent_path();

		if (modelPath == modelPath.root_path())
		{
			std::cout << "Unable to find Models: " << std::filesystem::current_path() << std::endl;
			std::cout << "ModelTest must be run from within the Utils subdirectory" << std::endl;

			return -1;
		}
	}

	modelPath = modelPath / "Models";

	std::cout << "Loading models from: " << modelPath << std::endl;

	std::cout << "WaveNet Test" << std::endl;
	RunTests(modelPath / "BossWN-standard.nam");

	std::cout << "LSTM Test" << std::endl;
	RunTests(modelPath / "BossLSTM-1x16.nam");

	return 0;
}

int main(int argc, char* argv[])
{
	if (RunDefaultTests() < 0)
		return -1;

	return 0;
}
