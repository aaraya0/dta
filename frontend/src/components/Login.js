import React , {useState} from 'react';
import httpClient from "../httpClient";
import "./login.css"
const Login: React.FC = () => {
    /** @type {string} */
    const [email, setEmail] = useState("");
    /** @type {string} */
    const [password, setPassword] = useState("");


    const logInUser = async () => {
        console.log(email,password);
        try{
        const resp= await httpClient.post("//localhost:5000/login", {
        email,
        password,
        });
        window.location.href="/";
        } catch (error){
        if(error.response.status === 401){
        alert("Invalid credentials");
        }}
    };


return (
<div>
    <h1>Log into your Account</h1>
    <form>
    <div>
    <label>Email:</label>
        <input type="text" value={email}
        onChange = {(e)=> setEmail(e.target.value)} id=""/>
    </div>
        <div>
    <label>Password:</label>
        <input type="password" value={password}
        onChange = {(e)=> setPassword(e.target.value)} id=""/>
    </div>
    <button type="button" onClick={()=> logInUser()}>Submit</button>
    </form>
</div>

);
}
export default Login