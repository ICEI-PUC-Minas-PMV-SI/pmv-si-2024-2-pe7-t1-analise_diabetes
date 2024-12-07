import React, { useState } from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  Input,
  Select,
  Button,
  useDisclosure,
  Text,
  useToast,
} from '@chakra-ui/react';
import { SymptomCheckbox } from './SymptomCheckbox';
import { SuccessModal } from './SuccessModal';
import { SymptomData, Gender } from '../types/symptoms';
import { translations } from '../translations/pt-br';
import { submitSymptoms } from '../services/api';

const initialSymptoms: SymptomData = {
  age: 24,
  gender: 'male',
  polyuria: false,
  polydipsia: false,
  sudden_weight_loss: false,
  partial_paresis: false,
  polyphagia: false,
  irritability: false,
  alopecia: false,
  visual_blurring: false,
  weakness: false,
  muscle_stiffness: false,
  genital_thrush: false,
  obesity: false,
  delayed_healing: false,
  itching: false,
};

export const SymptomForm: React.FC = () => {
  const [formData, setFormData] = useState<SymptomData>(initialSymptoms);
  const [hasDiabetesRisk, setHasDiabetesRisk] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await submitSymptoms(formData);
      setHasDiabetesRisk(response.final_classification);
      onOpen();
    } catch (error) {
      toast({
        title: translations.error,
        description: translations.errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'age' ? parseInt(value) || 0 : value,
    }));
  };

  const handleCheckboxChange = (name: keyof SymptomData) => {
    setFormData(prev => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  return (
    <>
      <form onSubmit={handleSubmit}>
        <VStack spacing={4} align="stretch" width="100%" maxW="500px" mx="auto">
          <FormControl isRequired>
            <FormLabel>{translations.age}</FormLabel>
            <Input
              type="number"
              name="age"
              value={formData.age}
              onChange={handleInputChange}
              min={0}
              max={120}
            />
          </FormControl>

          <FormControl isRequired>
            <FormLabel>{translations.gender}</FormLabel>
            <Select name="gender" value={formData.gender} onChange={handleInputChange}>
              <option value="male">{translations.male}</option>
              <option value="female">{translations.female}</option>
              <option value="other">{translations.other}</option>
            </Select>
          </FormControl>

          <Text fontWeight="bold" mt={4}>{translations.symptoms}</Text>

          {Object.keys(initialSymptoms).map(symptom => {
            if (symptom !== 'age' && symptom !== 'gender') {
              return (
                <SymptomCheckbox
                  key={symptom}
                  name={symptom}
                  isChecked={formData[symptom as keyof SymptomData] as boolean}
                  onChange={() => handleCheckboxChange(symptom as keyof SymptomData)}
                />
              );
            }
            return null;
          })}

          <Button 
            type="submit" 
            colorScheme="blue" 
            mt={4}
            size="lg"
            _hover={{ transform: 'translateY(-2px)' }}
            transition="all 0.2s"
          >
            {translations.submit}
          </Button>
        </VStack>
      </form>

      <SuccessModal 
        isOpen={isOpen} 
        onClose={onClose} 
        hasDiabetesRisk={hasDiabetesRisk}
      />
    </>
  );
};