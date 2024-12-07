import { SymptomData } from '../types/symptoms';

interface ApiResponse {
  final_classification: boolean;
}

export const submitSymptoms = async (data: SymptomData): Promise<ApiResponse> => {
  const transformedData = {
    age: data.age,
    gender: data.gender === 'male' ? 1 : data.gender === 'female' ? 0 : 2,
    polyuria: data.polyuria ? 1 : 0,
    polydipsia: data.polydipsia ? 1 : 0,
    sudden_weight_loss: data.sudden_weight_loss ? 1 : 0,
    partial_paresis: data.partial_paresis ? 1 : 0,
    polyphagia: data.polyphagia ? 1 : 0,
    irritability: data.irritability ? 1 : 0,
    alopecia: data.alopecia ? 1 : 0,
    visual_blurring: data.visual_blurring ? 1 : 0,
    weakness: data.weakness ? 1 : 0,
    muscle_stiffness: data.muscle_stiffness ? 1 : 0,
    genital_thrush: data.genital_thrush ? 1 : 0,
    obesity: data.obesity ? 1 : 0,
    delayed_healing: data.delayed_healing ? 1 : 0,
    itching: data.itching ? 1 : 0,
  };

  try {
    const response = await fetch('https://pmv-si-2024-2-pe7-t1-analise-diabetes.onrender.com/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(transformedData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error submitting symptoms:', error);
    throw error;
  }
};