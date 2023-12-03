using System;
using System.IO;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class InputLayer
    {
        private Random random = new Random();

        private (double[], int)[] trainSet = new(double[], int)[100];
        public (double[], int)[] TrainSet { get => trainSet; set => trainSet = value; }

        public InputLayer(NetworkMode networkMode)
        {
            switch (networkMode)
            {
                case NetworkMode.Train:
                    // здесь написать код считывания обучающего мн-ва из файла и формирования массива trainSet

                    string[] trainSample = File.ReadAllLines(AppDomain.CurrentDomain.BaseDirectory + "trainSample.txt");

                    for (int i = 0; i < trainSample.Length; i++)
                    {
                        string[] sample = trainSample[i].Split(' ');

                        trainSet[i].Item2 = int.Parse(sample[0]);

                        double[] tmpArr = new double[sample.Length - 1];
                        for (int j = 1; j < sample.Length; j++)
                        {
                            tmpArr[j - 1] = double.Parse(sample[j]);
                        }

                        trainSet[i].Item1 = tmpArr;
                    }

                    // перетасовка обучающей выборки методом Фишера-Йетса
                    for (int n = trainSet.Length - 1; n >= 1; n--)
                    {
                        int j = random.Next(n + 1);
                        (double[], int) temp = trainSet[n];
                        trainSet[n] = trainSet[j];
                        trainSet[j] = temp;
                    }
                    break;
                case NetworkMode.Test:
                    break;
                case NetworkMode.Demo:
                    break;
            }
        }

    }
}
