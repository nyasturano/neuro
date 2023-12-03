using System;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class NeuroNet
    {
        // массив для хранения вектора выходного сигнала нейросети
        public double[] Fact = new double[10];

        // все слои нейросети
        private InputLayer inputLayer = null;
        private HiddenLayer hiddenLayer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hiddenLayer1));
        private HiddenLayer hiddenLayer2 = new HiddenLayer(32, 71, NeuronType.Hidden, nameof(hiddenLayer2));
        private OutputLayer outputLayer = new OutputLayer(10, 32, NeuronType.Output, nameof(outputLayer));

        // среднее значение энергии ошибки эпохи обучения
        private double _eErrorAvg;

        // свойства
        public double EErrorAvg { get => _eErrorAvg; set => _eErrorAvg = value; }

        // конструктор
        public NeuroNet(NetworkMode networkMode)
        {
            inputLayer = new InputLayer(networkMode);
        }

        // прямой проход сигнала по нейросети
        public void ForwardPass(NeuroNet net, double[] netInput)
        {
            net.hiddenLayer1.Data = netInput;
            net.hiddenLayer1.Recognize(null, net.hiddenLayer2);
            net.hiddenLayer2.Recognize(null, net.outputLayer);
            net.outputLayer.Recognize(net, null);
        }

        // метод обучения
        public void Train(NeuroNet net)
        {
            // количество эпох обучения
            int epochs = 100;
            
            net.inputLayer = new InputLayer(NetworkMode.Train);

            double tmpSumError;
            // вектор сигнала ошибки входного слоя
            double[] errors;

            // вектор градиента 1 скрытого слоя
            double[] tmp_g_sums1;
            // вектор градиента 2 скрытого слоя
            double[] tmp_g_sums2;

            for (int k = 0; k < epochs; k++)
            {
                // в начале каждой эпохи значение средней энергии ошибки обнуляется
                _eErrorAvg = 0;

                for (int i = 0; i < net.inputLayer.TrainSet.Length; i++)
                {
                    // прямой проход
                    ForwardPass(net, net.inputLayer.TrainSet[i].Item1);

                    // вычисление ошибки по итерации
                    tmpSumError = 0;
                    errors = new double[net.Fact.Length];

                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.inputLayer.TrainSet[i].Item2)
                        {
                            errors[x] = -(net.Fact[x] - 1.0d);
                        }
                        else
                        {
                            errors[x] = -net.Fact[x];
                        }

                        tmpSumError += errors[x] * errors[x] / 2;
                    }

                    // суммарное згачение энергии ошибки эпох
                    _eErrorAvg += tmpSumError / errors.Length;

                    // обратный проход и коррекция весов
                    tmp_g_sums2 = net.outputLayer.BackwardPass(errors);
                    tmp_g_sums1 = net.hiddenLayer2.BackwardPass(tmp_g_sums2);
                    net.hiddenLayer1.BackwardPass(tmp_g_sums1);
                }

                // среднее значение энергии ошибки одной эпохи
                _eErrorAvg /= net.inputLayer.TrainSet.Length;

                // здесь написать код отображения среднего значения энергии ошибки эпохи на графике

            }

            // уборка входного слоя
            net.inputLayer = null;

            // сохранение скорректированных весов
            net.hiddenLayer1.WeightInitialize(MemoryMode.SET, nameof(hiddenLayer1) + "_memory.csv");
            net.hiddenLayer2.WeightInitialize(MemoryMode.SET, nameof(hiddenLayer2) + "_memory.csv");
            net.outputLayer.WeightInitialize(MemoryMode.SET, nameof(outputLayer) + "_memory.csv");
        }

    }
}
