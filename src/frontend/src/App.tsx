import React from 'react';
import { ChakraProvider, Box, Container, Heading, Text } from '@chakra-ui/react';
import { SymptomForm } from './components/SymptomForm';
import { translations } from './translations/pt-br';

function App() {
  return (
    <ChakraProvider>
      <Box minH="100vh" bg="gray.50" py={8}>
        <Container maxW="container.md">
          <Heading mb={2} textAlign="center">{translations.title}</Heading>
          <Text mb={6} textAlign="center" color="gray.600">
            {translations.subtitle}
          </Text>
          <SymptomForm />
        </Container>
      </Box>
    </ChakraProvider>
  );
}

export default App;