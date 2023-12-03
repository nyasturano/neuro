using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    public static class Utils
    {
        public static double Average(double[] arr)
        {
            return arr.Sum() / arr.Length;
        }

        public static double Dispersion(double[] arr)
        {
            double mean = Average(arr);

            double[] squaredDifferences = new double[arr.Length];

            for (int i = 0; i < arr.Length; i++)
            {
                squaredDifferences[i] = Math.Pow(arr[i] - mean, 2);
            }

            double dispersion = squaredDifferences.Sum() / squaredDifferences.Length;

            return dispersion;
        }
    }
}
