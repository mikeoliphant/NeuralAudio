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
			virtual size_t GetSize() const { return 0; }
			virtual size_t GetNumCols() const { return 0; }
			size_t GetNumChannels() const { return Channels; }
			virtual T* GetData() { return nullptr; }
			virtual T* GetData(size_t startCol)
			{
				(void)startCol;

				return nullptr;
			}
			virtual const T* GetDataConst() const { return nullptr; }
			virtual const T* GetDataConst(size_t startCol) const
			{
				(void)startCol;

				return nullptr;
			}
			virtual void SetZero() {}
	};

	template<typename T, int Channels, int Cols>
	class ChannelBuffer : public ChannelBufferBase<T, Channels>
	{
		public:
			ChannelBuffer()
			{
			}
			
			size_t GetSize() const override
			{
				return (size_t)(Channels * Cols);
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

			T& operator()(size_t row, size_t col)
			{
				return data[col][row];
			}

			const T& operator()(size_t row, size_t col) const
			{
				return data[col][row];
			}

			const ChannelRowSpan<T, Channels> Slice(size_t startCol, size_t numCols)
			{
				return ChannelRowSpan<T, Channels>(this, startCol, numCols);
			}

			const ChannelRowSpan<T, Channels> Slice(size_t numCols)
			{					
				return ChannelRowSpan<T, Channels>(this, numCols);
			}

			const Eigen::Map<Eigen::Matrix<T, Channels, Cols>> GetEigenMap()
			{
				return Eigen::Map<Eigen::Matrix<T, Channels, Cols>>(GetData(), Channels, GetNumCols());
			}

			const Eigen::Map<const Eigen::Matrix<T, Channels, Cols>> GetEigenMapConst() const
			{
				return Eigen::Map<const Eigen::Matrix<T, Channels, Cols>>(GetDataConst(), Channels, GetNumCols());
			}

		private:
			alignas(32) std::array<std::array<T, Channels>, Cols> data;
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

			const ChannelRowSpan<T, Channels> Slice(size_t startCol, size_t numCols) const
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol + startCol, numCols);
			}

			const ChannelRowSpan<T, Channels> Slice(size_t numCols) const
			{
				return ChannelRowSpan<T, Channels>(buffer, this->startCol, numCols);
			}

			size_t GetSize() const
			{
				return GetNumChannels() * GetNumCols();
			}

			size_t GetNumCols() const
			{
				return numCols;
			}

			size_t GetNumChannels() const
			{
				return Channels;
			}

			T* GetData() const
			{				
				return buffer->GetData(startCol);
			}

			const T* GetDataConst() const
			{
				return buffer->GetDataConst(startCol);
			}

			T* GetData(size_t startCol) const
			{
				return buffer->GetData(this->startCol + startCol);
			}

			const T* GetDataConst(size_t startCol) const
			{
				return buffer->GetDataConst(this->startCol + startCol);
			}

			Eigen::Map<Eigen::Matrix<T, Channels, Eigen::Dynamic>> GetEigenMap() const
			{
				return Eigen::Map<Eigen::Matrix<T, Channels, Eigen::Dynamic>>(GetData(), Channels, numCols);
			}

			const Eigen::Map<const Eigen::Matrix<T, Channels, Eigen::Dynamic>> GetEigenMapConst() const
			{
				return Eigen::Map<const Eigen::Matrix<T, Channels, Eigen::Dynamic>>(GetDataConst(), Channels, numCols);
			}

			void CopyData(const ChannelRowSpan<T, Channels>& srcSpan) const
			{
				const T* srcPtr = srcSpan.GetDataConst();
				T* destPtr = GetData();

				memmove(destPtr, srcPtr, Channels * numCols * sizeof(T));
			}

			void AddData(const ChannelRowSpan<T, Channels>& srcSpan) const
			{
				const T* srcPtr = srcSpan.GetDataConst();
				T* destPtr = GetData();

				for (size_t count = 0; count < Channels * numCols; count++)
				{
					*destPtr++ += *srcPtr++;
				}
			}

		private:
			ChannelBufferBase<T, Channels>* const buffer;
			const size_t startCol;
			const size_t numCols;
	};
}