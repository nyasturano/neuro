namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class OutputLayer : Layer
    {
        public OutputLayer(int non, int nonp, TypeNeuron nt, string nm_Layer) : base(non, nonp, nt, nm_Layer) { }

        public override void Recognize(NeuroNet net, Layer nextLayer)
        {
            double e_sum = 0;

            for (int i = 0; i < numofneurons; i++)
            {
                e_sum += Neurons[i].Output;
            }

            for (int i = 0; i < numofneurons; i++)
            {
                net.Fact[i] = Neurons[i].Output / e_sum;
            }
        }

        public override double[] BackwardPass(double[] errors)
        {
            double[] gr_sum = new double[numofprevneurons + 1];

            // вычисление градиетных сумм
            for (int j = 0; j < numofprevneurons + 1; j++)
            {
                double sum = 0;
                for (int k = 0; k < Neurons.Length; k++)
                {
                    sum += Neurons[k].Weights[j] * errors[k];
                }
                gr_sum[j] = sum;
            }

            // вычисление коррекции синаптических весов
            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double delta_w = 0;
                    // для коррекции порога
                    if (n == 0)
                    {
                        delta_w = momentum * lastdeltaweights[i, 0] + learningrate * errors[i];
                    }
                    // коррекция синаптических весов
                    else
                    {
                        delta_w = momentum * lastdeltaweights[i, n] + learningrate * Neurons[i].Inputs[n - 1] * errors[i];
                    }

                    lastdeltaweights[i, n] = delta_w;
                    Neurons[i].Weights[n] += delta_w; // коррекция весов
                }

            }
            return gr_sum;
        }

    }
}

