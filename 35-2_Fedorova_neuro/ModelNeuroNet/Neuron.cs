using static System.Math;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class Neuron
    {
        
        // поля
        private TypeNeuron type;
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
        public Neuron(double[] _weights, TypeNeuron _type)
        {
            weights = _weights;
            type = _type;
        }


        public void Activator()
        {
            double sum = weights[0];

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * weights[i + 1];
            }

            switch (type)
            {
                case TypeNeuron.Hidden:
                    output = LeakyReLU(sum);
                    derivative = LeakyReLU_derivative(sum);
                    break;
                case TypeNeuron.Output:
                    output = Exp(sum);
                    break;
            }

        }

        private double LeakyReLU(double arg)
        {
            return arg >= 0 ? arg : arg * alpha;
        }

        private double LeakyReLU_derivative(double arg)
        {
            return arg >= 0 ? 1 : alpha;
        }
    }
}
