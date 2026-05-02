import React, { useEffect, useState } from "react";
import axios from "axios";
import { Container, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, CircularProgress, AppBar, Toolbar } from "@mui/material";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

function App() {
  const [data, setData] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get("http://localhost:8000/api/data").then((res) => {
      setData(res.data);
      setLoading(false);
    });
    axios.get("http://localhost:8000/api/summary").then((res) => {
      setSummary(res.data);
    });
  }, []);

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Traffic Data Dashboard</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" style={{ marginTop: 32 }}>
        {loading ? (
          <CircularProgress />
        ) : (
          <>
            <Typography variant="h4" gutterBottom>
              Traffic Data Table
            </Typography>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {data.length > 0 &&
                      Object.keys(data[0]).map((col) => (
                        <TableCell key={col}>{col}</TableCell>
                      ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.slice(0, 20).map((row, idx) => (
                    <TableRow key={idx}>
                      {Object.values(row).map((val, i) => (
                        <TableCell key={i}>{val}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            <Typography variant="h5" style={{ marginTop: 32 }}>
              Data Summary
            </Typography>
            {summary && (
              <Paper style={{ padding: 16, marginBottom: 32 }}>
                <pre style={{ fontSize: 14 }}>{JSON.stringify(summary.describe, null, 2)}</pre>
              </Paper>
            )}
            <Typography variant="h5" style={{ marginTop: 32 }}>
              Sample Bar Chart
            </Typography>
            {data.length > 0 && (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.slice(0, 10)}>
                  <XAxis dataKey={Object.keys(data[0])[0]} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey={Object.keys(data[0])[1]} fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </>
        )}
      </Container>
    </>
  );
}

export default App;
