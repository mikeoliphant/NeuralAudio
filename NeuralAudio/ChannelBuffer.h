#pragma once

#include <cassert>
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
			virtual void SetZero(size_t startCol, size_t numCols) = 0;
			virtual T& operator()(size_t row, size_t col) = 0;

			virtual const T& operator()(size_t row, size_t col) const = 0;
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

			void SetZero(size_t startCol, size_t numCols) override
			{
				for (size_t col = 0; col < numCols; col++)
					data[startCol + col].fill(0);
			}

			T& operator()(size_t row, size_t col) override
			{
				return data[col][row];
			}

			const T& operator()(size_t row, size_t col) const override
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

			Eigen::Map<Eigen::Matrix<T, Cols, Channels, Eigen::RowMajor>> GetEigenMapTransposed()
			{
				return Eigen::Map<Eigen::Matrix<T, Cols, Channels, Eigen::RowMajor>>(GetData(), Channels, GetNumCols());
			}

			const Eigen::Map<const Eigen::Matrix<T, Cols, Channels, Eigen::RowMajor>> GetEigenMapTransposedConst() const
			{
				return Eigen::Map<const Eigen::Matrix<T, Cols, Channels, Eigen::RowMajor>>(GetDataConst(), Channels, GetNumCols());
			}

	private:
			alignas(32) std::array<std::array<T, Channels>, Cols> data;
	};


	template<typename T, int Channels>
	class ChannelBufferDynamic : public ChannelBufferBase<T, Channels>
	{
	public:
		ChannelBufferDynamic()
		{
			numCols = 0;
			data = nullptr;
		}

		ChannelBufferDynamic(T* data, size_t numCols) :
			data(data),
			numCols(numCols)
		{
		}

		size_t GetSize() const override
		{
			return (size_t)(Channels * numCols);
		}

		size_t GetNumCols() const override
		{
			return numCols;
		}

		T* GetData() override
		{
			return data;
		}

		const T* GetDataConst() const override
		{
			return data;
		}

		T* GetData(size_t startCol) override
		{
			return data + (startCol * Channels);
		}

		const T* GetDataConst(size_t startCol) const override
		{
			return data + (startCol * Channels);
		}

		void SetZero() override
		{
			std::fill(data, data + GetSize(), 0);
		}

		void SetZero(size_t startCol, size_t numCols) override
		{
			std::fill(data + (startCol * Channels), data + ((startCol + numCols) * Channels), 0);
		}

		T& operator()(size_t row, size_t col) override
		{
			return data[(col * Channels) + row];
		}

		const T& operator()(size_t row, size_t col) const override
		{
			return data[(col * Channels) + row];
		}

		const ChannelRowSpan<T, Channels> Slice(size_t startCol, size_t numCols)
		{
			return ChannelRowSpan<T, Channels>(this, startCol, numCols);
		}

		const ChannelRowSpan<T, Channels> Slice(size_t numCols)
		{
			return ChannelRowSpan<T, Channels>(this, numCols);
		}

		const Eigen::Map<Eigen::Matrix<T, Channels, Eigen::Dynamic>> GetEigenMap()
		{
			return Eigen::Map<Eigen::Matrix<T, Channels, Eigen::Dynamic>>(GetData(), Channels, GetNumCols());
		}

		const Eigen::Map<const Eigen::Matrix<T, Channels, Eigen::Dynamic>> GetEigenMapConst() const
		{
			return Eigen::Map<const Eigen::Matrix<T, Channels, Eigen::Dynamic>>(GetDataConst(), Channels, GetNumCols());
		}

		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>> GetEigenMapTransposed()
		{
			return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>>(GetData(), Channels, GetNumCols());
		}

		const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>> GetEigenMapTransposedConst() const
		{
			return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>>(GetDataConst(), Channels, GetNumCols());
		}

	private:
		T* data;
		size_t numCols;
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
				assert(startCol >= 0);
				assert(numCols <= (baseBuffer->GetNumCols() - startCol));
			}

			ChannelRowSpan(ChannelBufferBase<T, Channels>* baseBuffer, size_t numCols) :
				buffer(baseBuffer),
				startCol(0),
				numCols(numCols)
			{
				assert(numCols <= baseBuffer->GetNumCols());
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

			void SetZero() const
			{
				buffer->SetZero(startCol, numCols);
			}

			T& operator()(size_t row, size_t col)
			{
				return (*buffer)(row, startCol + col);
			}

			const T& operator()(size_t row, size_t col) const
			{
				return (*buffer)(row, startCol + col);
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

			Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>> GetEigenMapTransposed()
			{
				return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>>(GetData(), Channels, numCols);
			}

			const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>> GetEigenMapTransposedConst() const
			{
				return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Channels, Eigen::RowMajor>>(GetDataConst(), Channels, numCols);
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