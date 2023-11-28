namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class OutputLayer : Layer
    {
        public OutputLayer(int neuronsCount, int prevNeuronsCount, NeuronType neuronType, string name) 
            : base(neuronsCount, prevNeuronsCount, neuronType, name) { }

        public override void Recognize(NeuroNet net, Layer nextLayer)
        {
            double e_sum = 0;

            for (int i = 0; i < _neuronsCount; i++)
            {
                e_sum += Neurons[i].Output;
            }

            for (int i = 0; i < _neuronsCount; i++)
            {
                net.Fact[i] = Neurons[i].Output / e_sum;
            }
        }

        public override double[] BackwardPass(double[] errors)
        {
            double[] gr_sum = new double[_prevNeuronsCount + 1];

            // вычисление градиетных сумм
            for (int j = 0; j < _prevNeuronsCount + 1; j++)
            {
                double sum = 0;
                for (int k = 0; k < Neurons.Length; k++)
                {
                    sum += Neurons[k].Weights[j] * errors[k];
                }
                gr_sum[j] = sum;
            }

            // вычисление коррекции синаптических весов
            for (int i = 0; i < _neuronsCount; i++)
            {
                for (int n = 0; n < _prevNeuronsCount + 1; n++)
                {
                    double delta_w = 0;
                    // для коррекции порога
                    if (n == 0)
                    {
                        delta_w = _momentum * _lastDeltaWeights[i, 0] + _learningRate * errors[i];
                    }
                    // коррекция синаптических весов
                    else
                    {
                        delta_w = _momentum * _lastDeltaWeights[i, n] + _learningRate * Neurons[i].Inputs[n - 1] * errors[i];
                    }

                    _lastDeltaWeights[i, n] = delta_w;
                    Neurons[i].Weights[n] += delta_w; // коррекция весов
                }

            }
            return gr_sum;
        }

    }
}

