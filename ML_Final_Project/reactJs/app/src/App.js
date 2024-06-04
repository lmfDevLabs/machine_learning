import React from 'react';
import { Container, Typography } from '@mui/material';
import ProductSelector from './ProductSelector';

function App() {
  return (
    <Container>
      <Typography variant="h2" align="center" gutterBottom>
        Recommendation System
      </Typography>
      <ProductSelector />
    </Container>
  );
}

export default App;