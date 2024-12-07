import React from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  VStack,
  Text,
  Box,
} from '@chakra-ui/react';
import { AlertTriangle, CheckCircle } from 'lucide-react';
import { translations } from '../translations/pt-br';

interface SuccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  hasDiabetesRisk: boolean;
}

export const SuccessModal: React.FC<SuccessModalProps> = ({ 
  isOpen, 
  onClose, 
  hasDiabetesRisk 
}) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered motionPreset="scale">
      <ModalOverlay bg="blackAlpha.300" backdropFilter="blur(10px)" />
      <ModalContent mx={4}>
        <ModalHeader 
          display="flex" 
          alignItems="center" 
          justifyContent="center"
          pb={0}
        >
          {translations.success}
        </ModalHeader>
        <ModalCloseButton />
        <ModalBody pb={6}>
          <VStack spacing={4} py={4}>
            <Box color={hasDiabetesRisk ? "orange.500" : "green.500"}>
              {hasDiabetesRisk ? (
                <AlertTriangle size={48} strokeWidth={1.5} />
              ) : (
                <CheckCircle size={48} strokeWidth={1.5} />
              )}
            </Box>
            <Text 
              textAlign="center" 
              fontSize="lg"
              color="gray.700"
            >
              {hasDiabetesRisk 
                ? translations.visitDoctor 
                : translations.noSymptoms}
            </Text>
          </VStack>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};