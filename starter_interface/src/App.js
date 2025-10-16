import React from "react";
import { Container } from "@mui/material";
import VideoControl from "./components/VideoControl";
import LocationForm from "./components/LocationForm";
import TestSender from "./components/TestSender";

export default function App() {
  return (
    <Container maxWidth="sm" sx={{ mt: 5 }}>
      <VideoControl />
      <LocationForm />
      <TestSender />
    </Container>
  );
}
