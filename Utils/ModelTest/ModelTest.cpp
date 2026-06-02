#include <math.h>
#include <filesystem>
#include <iostream>
#include "argparse.hpp"
#include <NeuralAudio/NeuralModel.h>

using namespace NeuralAudio;

static std::string LoadModes[] = { "Internal", "RTNeural", "NAMCore" };

NeuralModel* LoadModel(std::filesystem::path modelPath, NeuralModelLoader& loader, EModelLoadMode loadMode)
{
	loader.SetWaveNetLoadMode(loadMode);
	loader.SetLSTMLoadMode(loadMode);

	if (!std::filesystem::exists(modelPath))
	{
		std::cout << "Model file does not exist: " << modelPath << std::endl;

		return nullptr;
	}

	try
	{
		auto model = loader.CreateFromFile(modelPath);

		if (model == nullptr)
		{
			std::cout << "Unable to load model from: " << modelPath << std::endl;

			return nullptr;
		}

		if (model->GetLoadMode() != loadMode)
		{
			delete model;

			return nullptr;
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
	catch (const std::exception& e)
	{
		std::cout << "Error loading model: " << e.what() << std::endl;
	}

	return nullptr;
}

static std::tuple<double, double> BenchModel(NeuralModel* model, int blockSize, int numBlocks)
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

	maxBlock *= numBlocks;

	return std::tie(tot, maxBlock);
}

static double ComputeError(NeuralModel* model1, NeuralModel* model2, int blockSize, int numBlocks)
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

void RunNAMTests(std::filesystem::path modelPath, NeuralModelLoader& loader, int blockSize)
{
	std::cout << "Model: " << modelPath << std::endl;
	std::cout << std::endl;

	int dataSize = 4096 * 64;

	int numBlocks = dataSize / blockSize;

	loader.SetDefaultMaxAudioBufferSize(blockSize);

	NeuralModel* rtNeuralModel = LoadModel(modelPath, loader, EModelLoadMode::RTNeural);
	NeuralModel* namCoreModel = LoadModel(modelPath, loader, EModelLoadMode::NAMCore);
	NeuralModel* internalModel = LoadModel(modelPath, loader, EModelLoadMode::Internal);

	double rms;

	std::tuple<double, double> internal;
	std::tuple<double, double> rtNeural;
	std::tuple<double, double> namCore;

	if (internalModel != nullptr)
	{
		internal = BenchModel(internalModel, blockSize, numBlocks);

		std::cout << "Internal: " << std::get<0>(internal) << " (" << std::get<1>(internal) << ")" << std::endl;
	}
	else
	{
		std::cout << "Model can't be loaded as internal model" << std::endl;
	}

	if (namCoreModel != nullptr)
	{
		std::cout << std::endl;

		namCore = BenchModel(namCoreModel, blockSize, numBlocks);

		std::cout << "NAM Core: " << std::get<0>(namCore) << " (" << std::get<1>(namCore) << ")" << std::endl;

		if (internalModel != nullptr)
		{
			rms = ComputeError(namCoreModel, internalModel, blockSize, numBlocks);

			std::cout << "NAM vs Internal RMS err: " << rms << std::endl;
			std::cout << "Internal is: " << (std::get<0>(namCore) / std::get<0>(internal)) << "x NAM" << std::endl;
		}
	}

	if (rtNeuralModel != nullptr)
	{
		std::cout << std::endl;

		rtNeural = BenchModel(rtNeuralModel, blockSize, numBlocks);

		std::cout << "RTNeural: " << std::get<0>(rtNeural) << " (" << std::get<1>(rtNeural) << ")" << std::endl;

		if (namCoreModel != nullptr)
		{
			rms = ComputeError(namCoreModel, rtNeuralModel, blockSize, numBlocks);
			std::cout << "NAM vs RTNeural RMS err: " << rms << std::endl;

			if (namCoreModel != nullptr)
			{
				std::cout << "RTNeural is: " << (std::get<0>(namCore) / std::get<0>(rtNeural)) << "x NAM" << std::endl;
			}
		}
	}

	std::cout << std::endl;
}

void RunKerasTests(std::filesystem::path modelPath, NeuralModelLoader& loader, int blockSize)
{
	std::cout << "Model: " << modelPath << std::endl;

	int dataSize = 4096 * 64;

	int numBlocks = dataSize / blockSize;

	loader.SetDefaultMaxAudioBufferSize(blockSize);

	auto internalModel = LoadModel(modelPath, loader, EModelLoadMode::Internal);
	auto rtNeuralModel = LoadModel(modelPath, loader, EModelLoadMode::RTNeural);

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

int RunDefaultTests(NeuralModelLoader& loader, int blockSize)
{
	std::filesystem::path modelPath = std::filesystem::current_path();

	do 
	{
		if (std::filesystem::exists((modelPath / "Models")))
			break;

		modelPath = modelPath.parent_path();
	}
	while (modelPath != modelPath.root_path());

	if (modelPath == modelPath.root_path())
	{
		std::cout << "Unable to find Models: " << std::filesystem::current_path() << std::endl;
		std::cout << "ModelTest looks for a \"Models\" folder in current folder or up the path." << std::endl;
		std::cout << "You can also specify a specific model to test by passing the path on the commandline." << std::endl;

		return -1;
	}

	modelPath = modelPath / "Models";

	std::cout << "Loading models from: " << modelPath << std::endl << std::endl;

	std::cout << "WaveNet (Standard) Test" << std::endl;
	RunNAMTests(modelPath / "BossWN-standard.nam", loader, blockSize);

	std::cout << std::endl;

	std::cout << "LSTM (1x16) Test" << std::endl;
	RunNAMTests(modelPath / "BossLSTM-1x16.nam", loader, blockSize);

	return 0;
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("ModelTest", "1.0.0");

	program.add_argument("model_file")
		.default_value("")
		.nargs(1)
        .help("Specify a specific model for testing");

	program.add_argument("-b", "--block_size")
		.default_value(64)
		.nargs(1)
		.required()
        .help("Specify the number of iterations")
        .scan<'i', int>();

	program.add_argument("-q", "--quality_scale")
		.default_value(1.0f)
		.nargs(1)
		.required()
		.help("Quality scaling (0.0 is fasteset, 1.0 is highest quality")
		.scan<'g', float>();
    try
	{
        program.parse_args(argc, argv);
    }
	catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

	std::cout << std::endl;

	int blockSize = 64;

	std::filesystem::path modelPath = R"(C:\Users\oliph\Downloads\This Boss SD-1 Does Not Exist\BossSD1LiveSPICE.nam)"; // program.get("model_file");

	blockSize = program.get<int>("--block_size");

	float qualityScale = 1.0f;

	qualityScale = program.get<float>("--quality_scale");

	NeuralModelLoader loader;

	loader.SetDefaultQualityScaleFactor(qualityScale);

	std::cout << "Block size: " << blockSize << "  Quality Scale: " << qualityScale << std::endl;

	if (!modelPath.empty())
	{
		if (modelPath.extension() == ".nam")
		{
			RunNAMTests(modelPath, loader, blockSize);
		}
		else
		{
			RunKerasTests(modelPath, loader, blockSize);
		}
	}
	else
	{
		if (RunDefaultTests(loader, blockSize) < 0)
			return -1;
	}

	return 0;
}
