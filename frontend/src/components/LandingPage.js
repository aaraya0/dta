import React, {useState, useEffect} from 'react';
import {User} from "../types";
import httpClient from "../httpClient";
import "./landing.css"
const LandingPage: React.FC = () =>{
    const [user, setUser]= useState(User)

    const logoutUser = async() => {
    const resp = await httpClient.post("//localhost:5000/logout")
    window.location.href = "/";
    }
    useEffect(()=> {
    (async () => {
        try{const resp = await httpClient.get("//localhost:5000/@me")
        setUser(resp.data)
        } catch(error){
        console.log("Not authenticated")}
     })();
    }, []);

    return (
    <div>
    <h1>Welcome</h1>
    {user != null ? (
    <div>
    <h2> HI, {user.name}! YOU ARE LOGGED IN </h2>
    <h3> Email: {user.email} </h3>
    <button onClick={logoutUser}>Logout</button>
    </div>
    ) : ( <div>
    <p>You are not logged in</p>
    <div className="button-span">
    <a href="/login"><button>Login</button></a>
    <a href="/register"><button>Register</button></a>
    </div>
    </div>)}

    </div>
    )
}

export default LandingPage