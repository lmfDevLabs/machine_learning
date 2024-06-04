import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Box, Button, FormControl, InputLabel, MenuItem, Select, Typography, List, ListItem, ListItemText } from '@mui/material';

const ProductSelector = () => {
  const [products, setProducts] = useState([]);
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [model, setModel] = useState('knn');
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8080/products')
      .then(response => {
        setProducts(response.data.products);
      })
      .catch(error => {
        console.error('There was an error fetching the product list!', error);
      });
  }, []);

  const handleProductChange = (event) => {
    setSelectedProducts(event.target.value);
  };

  const handleModelChange = (event) => {
    setModel(event.target.value);
  };

  const getRecommendations = () => {
    const apiUrl = model === 'knn' ? 'http://localhost:8080/knn' : 'http://localhost:8080/svm_lstm';
    axios.post(apiUrl, {
      product_names: selectedProducts,
      n_recommendations: 5
    })
      .then(response => {
        setRecommendations(response.data.recommended_products);
      })
      .catch(error => {
        console.error('There was an error fetching the recommendations!', error);
      });
  };

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        Product Selector
      </Typography>
      <FormControl fullWidth sx={{ mb: 4 }}>
        <InputLabel id="model-select-label">Select Model</InputLabel>
        <Select
          labelId="model-select-label"
          value={model}
          onChange={handleModelChange}
        >
          <MenuItem value="knn">KNN</MenuItem>
          <MenuItem value="svm_lstm">SVM-LSTM</MenuItem>
        </Select>
      </FormControl>
      <FormControl fullWidth sx={{ mb: 4 }}>
        <InputLabel id="product-select-label">Select Products</InputLabel>
        <Select
          labelId="product-select-label"
          multiple
          value={selectedProducts}
          onChange={handleProductChange}
        >
          {products.map((product, index) => (
            <MenuItem key={index} value={product}>{product}</MenuItem>
          ))}
        </Select>
      </FormControl>
      <Button variant="contained" color="primary" onClick={getRecommendations}>
        Get Recommendations
      </Button>
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5">
          Recommendations
        </Typography>
        <List>
          {recommendations.map((rec, index) => (
            <ListItem key={index}>
              <ListItemText primary={rec} />
            </ListItem>
          ))}
        </List>
      </Box>
    </Box>
  );
};

export default ProductSelector;