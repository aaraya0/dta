import React, { useState } from 'react';
import axios from 'axios';

function ExcelUploader() {
  const [file, setFile] = useState(null);

  const handleFileInputChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file', file);

    axios.post('http://127.0.0.1:5000/upload', formData)
      .then(response => {
        console.log(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Upload Excel file:
        <input type="file" onChange={handleFileInputChange} />
      </label>
      <button type="submit">Submit</button>
    </form>
  );
}

export default ExcelUploader;
