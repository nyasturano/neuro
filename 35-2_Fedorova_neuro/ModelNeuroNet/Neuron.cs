using static System.Math;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class Neuron
    {
        
        // поля
        private NeuronType _type;
        private double[] _inputs;
        private double[] _weights;

        private double _output;
        private double _derivative;

        // для LeakyReLU
        private double _alpha = 0.01;

        // свойства
        public double[] Inputs { get => _inputs; set => _inputs = value; }
        public double[] Weights { get => _weights; set => _weights = value; }
        
        public double Output { get => _output; }
        public double Derivative { get => _derivative; }

        // конструктор
        public Neuron(double[] weights, NeuronType type)
        {
            _weights = weights;
            _type = type;
        }


        public void Activator()
        {
            double sum = _weights[0];

            for (int i = 0; i < _inputs.Length; i++)
            {
                sum += _inputs[i] * _weights[i + 1];
            }

            switch (_type)
            {
                case NeuronType.Hidden:
                    _output = LeakyReLU(sum);
                    _derivative = LeakyReLU_derivative(sum);
                    break;
                case NeuronType.Output:
                    _output = Exp(sum);
                    break;
            }

        }

        private double LeakyReLU(double arg)
        {
            return arg >= 0 ? arg : arg * _alpha;
        }

        private double LeakyReLU_derivative(double arg)
        {
            return arg >= 0 ? 1 : _alpha;
        }
    }
}
