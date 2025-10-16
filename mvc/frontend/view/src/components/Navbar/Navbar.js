import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.css';
import './Navbar.css';
import logoImg from './havelsan-logo.png';

function Navbar() {
  return (
    <nav class="navbar navbar-expand-lg bg-body-tertiary" style={{padding: '0px'}}>
      <div class="container-fluid">
        <button class="navbar-toggler" style={{backgroundColor: 'white'}} type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarSupportedContent">
          <img src={logoImg}/>
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <li class="nav-item">
              {/* <a class="nav-link" href="/map">Map</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href='/logs'>Log</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href='/bounding-box'>Bounding Box</a> */}
            </li>
          </ul>
          
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
