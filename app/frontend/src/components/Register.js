import React , {useState} from 'react';
import httpClient from "../httpClient";
const Register: React.FC = () => {
    /** @type {string} */
    const [email, setEmail] = useState("");
    /** @type {string} */
    const [password, setPassword] = useState("");

    const [name, setName] = useState("");


    const registerUser = async () => {
        try{
        const resp= await httpClient.post("//localhost:5000/register", {
        email,
        password,
        name,
        });
        window.location.href="/";
        } catch (error){
        if(error.response.status === 401){
        alert("Invalid credentials");
        }}
    };


return (
<div>
    <h1>Create an Account</h1>
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
          <div>
    <label>Name:</label>
        <input type="text" value={name}
        onChange = {(e)=> setName(e.target.value)} id=""/>
    </div>

    <button type="button" onClick={()=> registerUser()}>Submit</button>
    </form>
</div>

);
}
export default Register