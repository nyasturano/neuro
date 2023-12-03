using System;

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

                    // перетасовка обучающей выборки методом Фишера-Йетса
                    break;
                case NetworkMode.Test:

                    break;
                case NetworkMode.Demo:
                    break;
            }
        }

    }
}
