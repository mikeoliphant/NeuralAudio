using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNet
{
    public static class VecOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot(ReadOnlySpan<float> vecA, ReadOnlySpan<float> vecB)
        {
            int offset = 0;
            float val = 0;

            if (vecA.Length >= Vector<float>.Count)
            {
                ReadOnlySpan<Vector<float>> aVecArray = MemoryMarshal.Cast<float, Vector<float>>(vecA);
                ReadOnlySpan<Vector<float>> bVecArray = MemoryMarshal.Cast<float, Vector<float>>(vecB);

                for (int i = 0; i < aVecArray.Length; i++)
                {
                    val += Vector.Dot(aVecArray[i], bVecArray[i]);
                }

                offset = aVecArray.Length * Vector<float>.Count;
            }

            for (int pos = offset; pos < vecA.Length; pos++)
            {
                val += vecA[pos] * vecB[pos];
            }

            return val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(Span<float> src, ReadOnlySpan<float> toAdd)
        {
            int offset = 0;

            if (src.Length >= Vector<float>.Count)
            {
                Span<Vector<float>> srcVecArray = MemoryMarshal.Cast<float, Vector<float>>(src);
                ReadOnlySpan<Vector<float>> toAddVecArray = MemoryMarshal.Cast<float, Vector<float>>(toAdd);

                for (int i = 0; i < srcVecArray.Length; i++)
                {
                    srcVecArray[i] += toAddVecArray[i];
                }

                offset = srcVecArray.Length * Vector<float>.Count;
            }

            for (int pos = offset; pos < toAdd.Length; pos++)
            {
                src[pos] += toAdd[pos];
            }
        }
    }

    public struct MatrixF
    {
        public int NumRows { get; private set; }
        public int NumCols { get; private set; }

        float[] data;

        public MatrixF(int numRows, int numCols)
        {
            this.NumRows = numRows;
            this.NumCols = numCols;

            data = new float[NumRows * NumCols];
        }

        public static MatrixF FromRowNormalData(ReadOnlySpan<float> data, int numRows, int numCols)
        {
            MatrixF newMatrix = new MatrixF(numRows, numCols);

            data.CopyTo(newMatrix.data);

            return newMatrix;
        }

        public override string ToString()
        {
            return "[" + NumRows + " x " + NumCols + "]";
        }

        public float this[int row, int col]
        {
            get
            {
                return data[(col * NumRows) + row];
            }

            set
            {
                data[(col * NumRows) + row] = value;
            }
        }

        public void Mult(ref MatrixF matB, ref MatrixF matOut)
        {
            for (int outRow = 0; outRow < matOut.NumRows; outRow++)
            {
                for (int outCol = 0; outCol < matOut.NumCols; outCol++)
                {
                    float val = 0;

                    for (int pos = 0; pos < NumCols; pos++)
                    {
                        val += data[(outRow * NumCols) + pos] * matB.data[(pos * matB.NumCols) + outCol];
                    }

                    matOut.data[(outRow * matOut.NumCols) + outCol] = val;
                }
            }
        }

        public void MultAcc(ref MatrixF matB, ref MatrixF matOut)
        {
            for (int outRow = 0; outRow < matOut.NumRows; outRow++)
            {
                for (int outCol = 0; outCol < matOut.NumCols; outCol++)
                {
                    float val = 0;

                    for (int pos = 0; pos < NumCols; pos++)
                    {
                        val += data[(outRow * NumCols) + pos] * matB.data[(pos * matB.NumCols) + outCol];
                    }

                    matOut.data[(outRow * matOut.NumCols) + outCol] += val;
                }
            }
        }

        public void Mult(ReadOnlySpan<float> vecB, Span<float> vecOut)
        {
            ReadOnlySpan<float> dataSpan = data;

            for (int outRow = 0; outRow < vecOut.Length; outRow++)
            {
                vecOut[outRow] = VecOp.Dot(dataSpan.Slice(outRow * vecB.Length, vecB.Length), vecB);
            }
        }

        public void MultAcc(ReadOnlySpan<float> vecB, Span<float> vecOut)
        {
            ReadOnlySpan<float> dataSpan = data;

            for (int outRow = 0; outRow < vecOut.Length; outRow++)
            {
                vecOut[outRow] += VecOp.Dot(dataSpan.Slice(outRow * vecB.Length, vecB.Length), vecB);
            }
        }

        public void Add(ref MatrixF toAdd)
        {
            if ((this.NumRows != toAdd.NumRows) || (this.NumCols != toAdd.NumCols))
            {
                throw new InvalidOperationException("Matrices must be the same size");
            }

            for (int pos = 0; pos < toAdd.data.Length; pos++)
            {
                this.data[pos] += toAdd.data[pos];
            }
        }
    }
}
