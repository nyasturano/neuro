using System;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class NeuroNet
    {
        // массив для хранения вектора выходного сигнала нейросети
        public double[] Fact = new double[10];

        // все слои нейросети
        private InputLayer input_Layer = null;
        private HiddenLayer hidden_Layer1 = new HiddenLayer(71, 15, TypeNeuron.Hidden, nameof(hidden_Layer1));
        private HiddenLayer hidden_Layer2 = new HiddenLayer(32, 71, TypeNeuron.Hidden, nameof(hidden_Layer2));
        private OutputLayer output_Layer = new OutputLayer(10, 32, TypeNeuron.Output, nameof(output_Layer));

        // среднее значение энергии ошибки эпохи обучения
        private double e_error_avr;

        // свойства
        public double E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        // конструктор
        public NeuroNet(NetworkMode nm)
        {
            input_Layer = new InputLayer(nm);
        }

        // прямой проход сигнала по нейросети
        public void ForwardPass(NeuroNet net, double[] netInput)
        {
            net.hidden_Layer1.Data = netInput;
            net.hidden_Layer1.Recognize(null, net.hidden_Layer2);
            net.hidden_Layer2.Recognize(null, net.output_Layer);
            net.output_Layer.Recognize(net, null);
        }

    }
}
