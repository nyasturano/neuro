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

    }
}
