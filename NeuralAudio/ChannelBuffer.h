#pragma once

#include <array>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace NeuralAudio
{
	template<typename T, int Channels>
	class ChannelRowSpan;

	template<typename T, int Channels>
	class ChannelBufferBase
	{
		public:
			virtual size_t GetNumCols() const { return 0; }
			size_t GetNumChannels() const { return Channels; }
			virtual T* GetData() { return nullptr; }
			virtual T* GetData(size_t startCol)
			{
				(void)startCol;

				return nullptr;
			}
			virtual const T* GetDataConst() const { return nullptr; }
			virtual const T* GetDataConst(size_t startCol) const { return nullptr; }
			virtual void SetZero() {}
	};

	template<typename T, int Channels, int Cols>
	class ChannelBuffer : public ChannelBufferBase<T, Channels>
	{
		public:
			ChannelBuffer()
			{
			}
			
			size_t GetNumCols() const override
			{
				return Cols;
			}

			T* GetData() override
			{
				return data[0].data();
			}

			const T* GetDataConst() const override
			{
				return data[0].data();
			}

			T* GetData(size_t startCol) override
			{
				return data[startCol].data();
			}

			const T* GetDataConst(size_t startCol) const override
			{
				return data[startCol].data();
			}

			void SetZero() override
			{
				for (auto& col : data) {
					col.fill(0);
				}
			}

			T& operator()(T row, T col)
			{
				return data[col][row];
			}

			const T& operator()(T row, T col) const
			{
				return data[col][row];
			}

			ChannelRowSpan<T, Channels> Slice(size_t startCol, size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(this, startCol, numCols);
			}

			const ChannelRowSpan<T, Channels> SliceConst(size_t startCol, size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(this, startCol, numCols);
			}

			ChannelRowSpan<T, Channels> Slice(size_t numCols)
			{					
				return ChannelRowSpan<T, Channels>(this, numCols);
			}

			const ChannelRowSpan<T, Channels> SliceConst(size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(this, numCols);
			}

			Eigen::Map<Eigen::Matrix<float, Channels, Cols>> GetEigenMap()
			{
				return Eigen::Map<Eigen::Matrix<float, Channels, Cols>>(GetData(), Channels, GetNumCols());
			}

			const Eigen::Map<const Eigen::Matrix<float, Channels, Cols>> GetEigenMapConst() const
			{
				return Eigen::Map<const Eigen::Matrix<float, Channels, Cols>>(GetDataConst(), Channels, GetNumCols());
			}

		private:
			std::array<std::array<T, Channels>, Cols> data;
	};

	template<typename T, int Channels>
	class ChannelRowSpan
	{
		public:
			ChannelRowSpan(ChannelBufferBase<T, Channels>& baseBuffer) :
				buffer(&baseBuffer),
				startCol(0),
				numCols(baseBuffer.GetNumCols())
			{
				
			}

			ChannelRowSpan(ChannelBufferBase<T, Channels>* baseBuffer) :
				buffer(baseBuffer),
				startCol(0),
				numCols(baseBuffer->GetNumCols())
			{
			
			}

			ChannelRowSpan(ChannelBufferBase<T, Channels>* baseBuffer, size_t startCol, size_t numCols) :
				buffer(baseBuffer),
				startCol(startCol),
				numCols(numCols)
			{

			}

			ChannelRowSpan(ChannelBufferBase<T, Channels>* baseBuffer, size_t numCols) :
				buffer(baseBuffer),
				startCol(0),
				numCols(numCols)
			{

			}

			ChannelRowSpan<T, Channels> Slice(size_t startCol, size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol + startCol, numCols);
			}

			const ChannelRowSpan<T, Channels> SliceConst(size_t startCol, size_t numCols) const
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol + startCol, numCols);
			}

			ChannelRowSpan<T, Channels> Slice(size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol, numCols);
			}

			const ChannelRowSpan<T, Channels> SliceConst(size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol, numCols);
			}

			size_t GetNumCols() const
			{
				return numCols;
			}

			size_t GetNumChannels() const
			{
				return Channels;
			}

			T* GetData()
			{				
				return buffer->GetData();
			}

			const T* GetDataConst() const
			{
				return buffer->GetDataConst();
			}

			T* GetData(size_t startCol)
			{
				return buffer->GetData(startCol);
			}

			const T* GetDataConst(size_t startCol) const
			{
				return buffer->GetDataConst(startCol);
			}

			Eigen::Map<Eigen::Matrix<float, Channels, Eigen::Dynamic>> GetEigenMap()
			{
				return Eigen::Map<Eigen::Matrix<float, Channels, Eigen::Dynamic>>(GetData(), Channels, numCols);
			}

			const Eigen::Map<const Eigen::Matrix<float, Channels, Eigen::Dynamic>> GetEigenMapConst() const
			{
				return Eigen::Map<const Eigen::Matrix<float, Channels, Eigen::Dynamic>>(GetDataConst(), Channels, numCols);
			}

			void CopyData(const ChannelRowSpan<T, Channels>& srcSpan)
			{
				const T* srcPtr = srcSpan.GetDataConst();
				T* destPtr = GetData();

				memcpy(destPtr, srcPtr, Channels * numCols);
			}

			void AddData(const ChannelRowSpan<T, Channels>& srcSpan)
			{
				const T* srcPtr = srcSpan.GetDataConst();
				T* destPtr = GetData();

				for (size_t count = 0; count < Channels * numCols; count++)
				{
					*destPtr++ += *srcPtr++;
				}
			}

		private:
			ChannelBufferBase<T, Channels>* buffer;
			const size_t startCol;
			const size_t numCols;
	};
}