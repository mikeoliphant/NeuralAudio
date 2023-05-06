using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNet
{
    public static class VecOp
    {
        public static float Dot(float[] vecA, float[] vecB)
        {
            float val = 0;

            for (int pos = 0; pos < vecA.Length; pos++)
            {
                val += vecA[pos] * vecB[pos];
            }

            return val;
        }

        public static void Add(float[] src, float[] toAdd)
        {
            for (int pos = 0; pos < src.Length; pos++)
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

        public static MatrixF FromRowNormalData(float[] data, int numRows, int numCols)
        {
            MatrixF newMatrix = new MatrixF(numRows, numCols);

            Array.Copy(data, newMatrix.data, data.Length);

            return newMatrix;
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

        public void Mult(float[] vecB, float[] vecOut)
        {
            int dataOffset = 0;

            for (int outRow = 0; outRow < vecOut.Length; outRow++)
            {
                float val = 0;

                for (int pos = 0; pos < vecB.Length; pos++)
                {
                    val += data[dataOffset++] * vecB[pos];
                }

                vecOut[outRow] = val;
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

        public void MultAcc(float[] vecB, float[] vecOut)
        {
            int dataOffset = 0;

            for (int outRow = 0; outRow < vecOut.Length; outRow++)
            {
                float val = 0;

                for (int pos = 0; pos < vecB.Length; pos++)
                {
                    val += data[dataOffset++] * vecB[pos];
                }

                vecOut[outRow] += val;
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
