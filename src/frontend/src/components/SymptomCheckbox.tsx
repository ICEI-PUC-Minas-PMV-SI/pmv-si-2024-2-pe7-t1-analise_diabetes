import React from 'react';
import { FormControl, Checkbox, Tooltip } from '@chakra-ui/react';
import { InfoIcon } from 'lucide-react';
import { symptomDescriptions } from '../translations/pt-br';

interface SymptomCheckboxProps {
  name: string;
  isChecked: boolean;
  onChange: () => void;
}

export const SymptomCheckbox: React.FC<SymptomCheckboxProps> = ({ name, isChecked, onChange }) => {
  const symptomInfo = symptomDescriptions[name as keyof typeof symptomDescriptions];

  return (
    <FormControl display="flex" alignItems="center">
      <Checkbox isChecked={isChecked} onChange={onChange}>
        {symptomInfo.name}
      </Checkbox>
      <Tooltip label={symptomInfo.description} hasArrow placement="right">
        <InfoIcon size={16} style={{ marginLeft: '8px', cursor: 'help' }} />
      </Tooltip>
    </FormControl>
  );
};