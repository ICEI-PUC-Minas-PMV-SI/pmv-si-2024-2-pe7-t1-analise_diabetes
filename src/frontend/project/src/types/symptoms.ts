export type Gender = 'male' | 'female' | 'other';

export interface SymptomData {
  age: number;
  gender: Gender;
  polyuria: boolean;
  polydipsia: boolean;
  sudden_weight_loss: boolean;
  partial_paresis: boolean;
  polyphagia: boolean;
  irritability: boolean;
  alopecia: boolean;
  visual_blurring: boolean;
  weakness: boolean;
  muscle_stiffness: boolean;
  genital_thrush: boolean;
  obesity: boolean;
  delayed_healing: boolean;
  itching: boolean;
}