using static System.Math;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class Neuron
    {   
        // поля
        private NeuronType type;
        private double[] inputs;
        private double[] weights;

        private double output;
        private double derivative;

        // для LeakyReLU
        private double alpha = 0.01;

        // свойства
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double[] Weights { get => weights; set => weights = value; }
        
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // конструктор
        public Neuron(double[] _weights, NeuronType _type)
        {
            weights = _weights;
            type = _type;
        }

        public void Activator()
        {
            // порог
            double sum = weights[0];

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * weights[i + 1];
            }

            switch (type)
            {
                case NeuronType.Hidden:
                    output = LeakyReLU(sum);
                    derivative = LeakyReLUderivative(sum);
                    break;
                case NeuronType.Output:
                    output = Exp(sum);
                    break;
            }
        }

        private double LeakyReLU(double arg)
        {
            return arg >= 0 ? arg : arg * alpha;
        }

        private double LeakyReLUderivative(double arg)
        {
            return arg >= 0 ? 1 : alpha;
        }
    }
}
