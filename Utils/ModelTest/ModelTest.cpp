#include <filesystem>
#include <iostream>
#include <NeuralAudio/NeuralModel.h>

static std::string LoadModes[] = { "Internal", "RTNeural", "NAMCore" };

NeuralAudio::NeuralModel* LoadModel(std::filesystem::path modelPath, NeuralAudio::EModelLoadMode loadMode)
{
	NeuralAudio::NeuralModel::SetWaveNetLoadMode(loadMode);
	NeuralAudio::NeuralModel::SetLSTMLoadMode(loadMode);

	try
	{
		auto model = NeuralAudio::NeuralModel::CreateFromFile(modelPath);

		if (model == nullptr)
		{
			std::cout << "Unable to load model from: " << modelPath << std::endl;

			return nullptr;
		}

		if (model->GetLoadMode() != loadMode)
		{
			std::cout << "**Warning: Tried to load " << LoadModes[loadMode] << " but got " << LoadModes[model->GetLoadMode()] << std::endl;
		}

		if (model->GetLoadMode() != NeuralAudio::EModelLoadMode::NAMCore)
		{
			if (!model->IsStatic())
			{
				std::cout << "**Warning: " << LoadModes[model->GetLoadMode()] << " model is not using a static architecture" << std::endl;
			}
		}

		return model;
	}
	catch (...)
	{
		std::cout << "Error loading model" << std::endl;
	}

	return nullptr;
}

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

	model1->Prewarm();
	model2->Prewarm();

	double totErr = 0;

	long pos = 0;

	for (int block = 0; block < numBlocks; block++)
	{
		for (int i = 0; i < blockSize; i++)
		{
			inData[i] = (float)sin(pos++ * 0.01);
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

void RunNAMTests(std::filesystem::path modelPath, int blockSize)
{
	std::cout << "Model: " << modelPath << std::endl;
	std::cout << std::endl;

	int dataSize = 4096 * 64;

	int numBlocks = dataSize / blockSize;

	NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(blockSize);

	auto rtNeuralModel = LoadModel(modelPath, NeuralAudio::EModelLoadMode::RTNeural);
	auto namCoreModel = LoadModel(modelPath, NeuralAudio::EModelLoadMode::NAMCore);
	auto internalModel = LoadModel(modelPath, NeuralAudio::EModelLoadMode::Internal);

	double rms = ComputeError(namCoreModel, internalModel, blockSize, numBlocks);
	std::cout << "NAM vs Internal RMS err: " << rms << std::endl;

	rms = ComputeError(namCoreModel, rtNeuralModel, blockSize, numBlocks);
	std::cout << "NAM vs RTNeural RMS err: " << rms << std::endl;
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

	std::cout << "***here" << std::endl;
}

void RunKerasTests(std::filesystem::path modelPath, int blockSize)
{
	std::cout << "Model: " << modelPath << std::endl;

	int dataSize = 4096 * 64;

	int numBlocks = dataSize / blockSize;

	NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(blockSize);

	auto internalModel = LoadModel(modelPath, NeuralAudio::EModelLoadMode::Internal);
	auto rtNeuralModel = LoadModel(modelPath, NeuralAudio::EModelLoadMode::RTNeural);

	double rms = ComputeError(rtNeuralModel, internalModel, blockSize, numBlocks);
	std::cout << "Internal vs RTNeural RMS err: " << rms << std::endl;
	std::cout << std::endl;

	auto internal = BenchModel(internalModel, blockSize, numBlocks);
	auto rt = BenchModel(rtNeuralModel, blockSize, numBlocks);

	std::cout << "RTNeural: " << std::get<0>(rt) << " (" << std::get<1>(rt) << ")" << std::endl;
	std::cout << "Internal: " << std::get<0>(internal) << " (" << std::get<1>(internal) << ")" << std::endl;
	std::cout << "Internal is: " << (std::get<0>(rt) / std::get<0>(internal)) << "x RTNeural" << std::endl;

	std::cout << std::endl;
}

int RunDefaultTests(int blockSize)
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

	std::cout << "WaveNet (Standard) Test" << std::endl;
	RunNAMTests(modelPath / "BossWN-standard.nam", blockSize);

	std::cout << "LSTM (1x16) Test" << std::endl;
	RunNAMTests(modelPath / "BossLSTM-1x16.nam", blockSize);

	return 0;
}

int main(int argc, char* argv[])
{
	int blockSize = 64;

	std::filesystem::path modelPath;

	for (int arg = 1; arg < argc; arg++)
	{
		char* end;

		long val = strtol(argv[arg], &end, 10);

		if (val != 0)
		{
			blockSize = (int)val;
		}
		else
		{
			modelPath.assign(argv[arg]);
		}
	}

	std::cout << "Block size: " << blockSize << std::endl;

	if (!modelPath.empty())
	{
		if (modelPath.extension() == ".nam")
		{
			RunNAMTests(modelPath, blockSize);
		}
		else
		{
			RunKerasTests(modelPath, blockSize);
		}
	}
	else
	{
		if (RunDefaultTests(blockSize) < 0)
			return -1;
	}

	return 0;
}
