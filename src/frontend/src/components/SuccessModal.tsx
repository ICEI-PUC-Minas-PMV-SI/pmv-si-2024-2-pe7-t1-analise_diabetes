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
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
} from '@chakra-ui/react';
import { AlertTriangle, CheckCircle } from 'lucide-react';
import { translations } from '../translations/pt-br';

interface SuccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  hasDiabetesRisk: boolean;
  predictions?: {
    forest: boolean;
    naive: boolean;
    tree: boolean;
  };
}

export const SuccessModal: React.FC<SuccessModalProps> = ({ 
  isOpen, 
  onClose, 
  hasDiabetesRisk,
  predictions 
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

            {predictions && (
              <Box w="100%" mt={4}>
                <Text fontWeight="bold" mb={2}>{translations.modelPredictions}</Text>
                <Table size="sm" variant="simple">
                  <Thead>
                    <Tr>
                      <Th>{translations.model}</Th>
                      <Th>{translations.prediction}</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    <Tr>
                      <Td>Random Forest</Td>
                      <Td color={predictions.forest ? "orange.500" : "green.500"}>
                        {predictions.forest ? translations.positive : translations.negative}
                      </Td>
                    </Tr>
                    <Tr>
                      <Td>Naive Bayes</Td>
                      <Td color={predictions.naive ? "orange.500" : "green.500"}>
                        {predictions.naive ? translations.positive : translations.negative}
                      </Td>
                    </Tr>
                    <Tr>
                      <Td>Decision Tree</Td>
                      <Td color={predictions.tree ? "orange.500" : "green.500"}>
                        {predictions.tree ? translations.positive : translations.negative}
                      </Td>
                    </Tr>
                  </Tbody>
                </Table>
              </Box>
            )}
          </VStack>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};