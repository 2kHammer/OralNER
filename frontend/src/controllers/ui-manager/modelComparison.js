const data = [
    { id: 1, name: 'Eintrag 1', value: 'Wert 1' },
    { id: 2, name: 'Eintrag 2', value: 'Wert 2' },
    { id: 3, name: 'Eintrag 3', value: 'Wert 3' },
    // mehr Daten ...
  ];
  
  function createTable(data) {
    const container = document.getElementById('table-container');
    
    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';
    
    // Kopfzeile
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['ID', 'Name', 'Wert'].forEach(text => {
      const th = document.createElement('th');
      th.innerText = text;
      th.style.border = '1px solid #ccc';
      th.style.padding = '8px';
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
  
    // Körper
    const tbody = document.createElement('tbody');
    data.forEach(item => {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
  
      tr.addEventListener('click', () => {
        // Entferne Auswahl bei allen
        tbody.querySelectorAll('tr').forEach(row => row.style.backgroundColor = '');
        // Markiere ausgewählte Zeile
        tr.style.backgroundColor = '#cce5ff';
        console.log('Ausgewählte ID:', item.id);
      });
  
      Object.values(item).forEach(value => {
        const td = document.createElement('td');
        td.innerText = value;
        td.style.border = '1px solid #ccc';
        td.style.padding = '8px';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  
    container.innerHTML = ''; // alten Inhalt löschen
    container.appendChild(table);
  }
  
  createTable(data);
  