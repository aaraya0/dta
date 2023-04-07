import { BrowserRouter, Routes, Route} from "react-router-dom";
import LandingPage from "./components/LandingPage";
import Login from "./components/Login";
import Register from "./components/Register";
import NotFound from "./components/NotFound";
import ExcelUploader from "./components/ExcelUploader";
const Router = () => {
    return (
    <BrowserRouter>
    <Routes>
    <Route exact path="/" element={<LandingPage/>}/>
     <Route exact path="/login" element={<Login/>}/>
      <Route exact path="/register" element={<Register/>}/>
      <Route exact path="/upload" element={<ExcelUploader/>}/>
      <Route element={<NotFound/>}/>
    </Routes>
    </BrowserRouter>

    )
}
export default Router